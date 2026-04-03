"""
scripts/adapters/yolo_adapter.py
---------------------------------
Adapter for datasets already in YOLO detection format.

Input structure (standard YOLO):
    source_dir/
    ├── data.yaml               (optional — we parse it if it exists)
    ├── train/
    │   ├── images/
    │   │   └── *.jpg
    │   └── labels/
    │       └── *.txt
    └── valid/  (or val/)
        ├── images/
        └── labels/

Alternative flat structure:
    source_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Each label .txt file contains lines:
    <class_id> <x_center> <y_center> <width> <height>

This adapter:
    1. Discovers the directory layout (handles both common YOLO structures)
    2. Reads the source data.yaml to learn native class names
    3. Remaps native class IDs → master Krishi IDs using class_map
    4. Copies images and translated labels to staging
    5. Drops any annotations whose class is not in class_map
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from scripts.adapters.base import BaseAdapter

logger = logging.getLogger("krishi.adapter.yolo")


class YOLOAdapter(BaseAdapter):
    """Processes YOLO-format detection datasets with class ID remapping."""

    def process(
        self,
        source_dir: Path,
        staging_dir: Path,
        class_map: Dict[str, int],
        source_name: str,
        split_strategy: str = "preserve",
        val_ratio: float = 0.2,
    ) -> Dict[str, int]:

        self.ensure_staging_dirs(staging_dir)
        lookup = self.build_class_map_lookup(class_map)

        # 1. Discover native class names from data.yaml (if present)
        native_names = self._load_native_classes(source_dir)

        # 2. Build native_id → master_id remapping table
        id_remap = self._build_id_remap(native_names, lookup)

        # 3. Discover directory layout
        splits = self._discover_splits(source_dir, split_strategy)

        stats = {
            "images_processed": 0,
            "labels_written": 0,
            "classes_found": {},
            "unmapped_classes": [],
            "errors": 0,
        }

        all_items: List[Tuple[Path, Path, str]] = []  # (img, lbl, split)

        for split_name, (img_dir, lbl_dir) in splits.items():
            if img_dir is None or not img_dir.exists():
                logger.warning(f"[{source_name}] Split '{split_name}' image dir not found, skipping")
                continue

            images = sorted(
                [f for f in img_dir.iterdir() if self.is_image_file(f)]
            )

            for img_path in images:
                label_path = self._find_label(img_path, lbl_dir)
                all_items.append((img_path, label_path, split_name))

        # If auto-split is requested and we only have one split, redistribute
        if split_strategy == "auto" or (
            split_strategy == "preserve"
            and all(s == "train" for _, _, s in all_items)
            and len(all_items) > 0
        ):
            random.seed(42)
            random.shuffle(all_items)
            n_val = max(1, int(len(all_items) * val_ratio))
            reassigned = []
            for i, (img, lbl, _) in enumerate(all_items):
                new_split = "val" if i < n_val else "train"
                reassigned.append((img, lbl, new_split))
            all_items = reassigned
            logger.info(
                f"[{source_name}] Auto-split: {len(all_items) - n_val} train / {n_val} val"
            )

        # Process each image-label pair
        for img_path, label_path, split in all_items:
            try:
                new_name = self.build_output_name(source_name, img_path, source_dir)
                label_name = Path(new_name).with_suffix(".txt").name

                # Remap label
                remapped_lines = []
                if label_path is not None and label_path.exists():
                    remapped_lines = self._remap_label_file(
                        label_path, id_remap, native_names, lookup, stats
                    )

                # Only copy if we have at least one valid annotation
                if remapped_lines:
                    self.copy_image(img_path, staging_dir, split, new_name)
                    self.write_label(staging_dir, split, label_name, remapped_lines)
                    stats["images_processed"] += 1
                    stats["labels_written"] += 1
                else:
                    # Image with no mappable annotations — skip
                    stats["errors"] += 1

            except Exception as e:
                logger.error(f"[{source_name}] Error processing {img_path.name}: {e}")
                stats["errors"] += 1

        logger.info(
            f"[{source_name}] Complete: "
            f"{stats['images_processed']} images, "
            f"{stats['labels_written']} labels, "
            f"{stats['errors']} errors"
        )
        if stats["unmapped_classes"]:
            logger.warning(
                f"[{source_name}] Unmapped classes encountered: "
                f"{list(set(stats['unmapped_classes']))}"
            )

        return stats

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _load_native_classes(self, source_dir: Path) -> Dict[int, str]:
        """
        Parse data.yaml to get {native_id: class_name} mapping.
        Returns empty dict if no data.yaml found.
        """
        candidates = [
            source_dir / "data.yaml",
            source_dir / "data.yml",
            source_dir / "dataset.yaml",
        ]

        for yaml_path in candidates:
            if yaml_path.exists():
                try:
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        cfg = yaml.safe_load(f)

                    names = cfg.get("names", {})
                    if isinstance(names, list):
                        return {i: n for i, n in enumerate(names)}
                    elif isinstance(names, dict):
                        return {int(k): v for k, v in names.items()}
                except Exception as e:
                    logger.warning(f"Failed to parse {yaml_path}: {e}")

        logger.info(f"No data.yaml found in {source_dir}, using raw class IDs")
        return {}

    def _build_id_remap(
        self,
        native_names: Dict[int, str],
        lookup: Dict[str, int],
    ) -> Dict[int, Optional[int]]:
        """
        Build {native_class_id: master_class_id} mapping.

        If native_names is empty (no data.yaml), we try to match
        the raw integer IDs as string keys in the lookup.
        """
        remap = {}

        if native_names:
            for native_id, native_name in native_names.items():
                key = native_name.lower().strip()
                master_id = lookup.get(key)
                if master_id is None:
                    # Try without underscores/spaces
                    master_id = lookup.get(key.replace("_", " "))
                if master_id is None:
                    master_id = lookup.get(key.replace(" ", "_"))
                remap[native_id] = master_id
        else:
            # No data.yaml — try direct integer matching
            for key, master_id in lookup.items():
                try:
                    remap[int(key)] = master_id
                except ValueError:
                    pass

        return remap

    def _remap_label_file(
        self,
        label_path: Path,
        id_remap: Dict[int, Optional[int]],
        native_names: Dict[int, str],
        lookup: Dict[str, int],
        stats: Dict,
    ) -> List[str]:
        """
        Read a YOLO label file and remap class IDs.
        Returns list of remapped lines (drops unmapped classes).
        """
        remapped = []

        for line in label_path.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            try:
                native_id = int(parts[0])
            except ValueError:
                continue  # Skip malformed lines

            # Try the pre-built remap table first
            master_id = id_remap.get(native_id)

            if master_id is None and native_id not in id_remap:
                # Dynamic fallback: try matching the native name
                native_name = native_names.get(native_id, str(native_id))
                key = native_name.lower().strip()
                master_id = lookup.get(key) or lookup.get(
                    key.replace("_", " ")
                ) or lookup.get(key.replace(" ", "_"))
                # Cache for future lookups
                id_remap[native_id] = master_id

            if master_id is None:
                # Unmapped class — track and skip
                native_name = native_names.get(native_id, str(native_id))
                stats["unmapped_classes"].append(native_name)
                continue

            # Track class distribution
            stats["classes_found"][master_id] = (
                stats["classes_found"].get(master_id, 0) + 1
            )

            # Rebuild line with remapped class ID
            remapped.append(f"{master_id} {' '.join(parts[1:])}")

        return remapped

    def _discover_splits(
        self, source_dir: Path, split_strategy: str
    ) -> Dict[str, Tuple[Optional[Path], Optional[Path]]]:
        """
        Discover the train/val directory structure.
        Returns {split_name: (images_dir, labels_dir)}.

        Handles multiple common YOLO directory layouts.
        """
        splits = {}

        # Layout A: source_dir/images/{train,val}/ + source_dir/labels/{train,val}/
        for split_name in ("train", "val", "valid", "test"):
            img_dir = source_dir / "images" / split_name
            lbl_dir = source_dir / "labels" / split_name

            if img_dir.exists():
                actual_split = "val" if split_name in ("valid", "val") else split_name
                splits[actual_split] = (img_dir, lbl_dir if lbl_dir.exists() else None)

        if splits:
            return splits

        # Layout B: source_dir/{train,valid}/{images,labels}/
        for split_name in ("train", "val", "valid", "test"):
            split_dir = source_dir / split_name
            if split_dir.exists():
                img_dir = split_dir / "images"
                lbl_dir = split_dir / "labels"
                if img_dir.exists():
                    actual_split = "val" if split_name in ("valid", "val") else split_name
                    splits[actual_split] = (img_dir, lbl_dir if lbl_dir.exists() else None)

        if splits:
            return splits

        # Layout C: Flat — source_dir/images/ + source_dir/labels/ (single split)
        img_dir = source_dir / "images"
        lbl_dir = source_dir / "labels"
        if img_dir.exists():
            splits["train"] = (img_dir, lbl_dir if lbl_dir.exists() else None)
            return splits

        # Layout D: Everything in root (images and labels mixed)
        images = [f for f in source_dir.iterdir() if self.is_image_file(f)]
        if images:
            splits["train"] = (source_dir, source_dir)
            return splits

        logger.error(f"Could not discover any valid layout in {source_dir}")
        return {}

    @staticmethod
    def _find_label(img_path: Path, lbl_dir: Optional[Path]) -> Optional[Path]:
        """Find the label file corresponding to an image."""
        if lbl_dir is None:
            # Try same directory as image
            lbl_path = img_path.with_suffix(".txt")
            return lbl_path if lbl_path.exists() else None

        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        return lbl_path if lbl_path.exists() else None
