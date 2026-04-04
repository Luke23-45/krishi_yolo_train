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
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import yaml

from scripts.adapters.base import BaseAdapter, CanonicalObject, CanonicalSample

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
        default_master_id = self.infer_single_target_class(class_map)

        # 1. Discover native class names from data.yaml (if present)
        native_names = self._load_native_classes(source_dir)

        # 2. Build native_id → master_id remapping table
        id_remap = self._build_id_remap(native_names, lookup, default_master_id)

        # 3. Discover directory layout
        splits = self._discover_splits(source_dir, split_strategy)

        stats = {
            "images_processed": 0,
            "labels_written": 0,
            "classes_found": {},
            "unmapped_classes": [],
            "auto_mapped_classes": [],
            "samples": [],
            "errors": 0,
        }

        all_items: List[Tuple[Path, Optional[Path], str]] = []  # (img, lbl, split)

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

        if not all_items:
            all_items = self._discover_items_recursively(source_dir, source_name)

        # If auto-split is requested and we only have one split, redistribute
        should_autosplit = (
            split_strategy == "auto"
            or (
                split_strategy == "preserve"
                and all(s == "train" for _, _, s in all_items)
            )
        )
        if all_items and should_autosplit:
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
                width, height = self.get_image_size(img_path)

                # Remap label
                remapped_objects: List[CanonicalObject] = []
                if label_path is not None and label_path.exists():
                    remapped_objects = self._remap_label_file(
                        label_path,
                        id_remap,
                        native_names,
                        lookup,
                        stats,
                        default_master_id,
                        width,
                        height,
                    )

                # Only copy if we have at least one valid annotation
                if remapped_objects:
                    sample = CanonicalSample(
                        image_id=Path(new_name).stem,
                        image_path=str(img_path.resolve()),
                        output_file_name=new_name,
                        width=width,
                        height=height,
                        split=split,
                        objects=remapped_objects,
                        source_name=source_name,
                    )
                    stats["samples"].append(sample)
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
        if stats["auto_mapped_classes"]:
            logger.info(
                f"[{source_name}] Auto-mapped variant classes: "
                f"{list(set(stats['auto_mapped_classes']))}"
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
        candidates.extend(sorted(source_dir.rglob("data.yaml")))
        candidates.extend(sorted(source_dir.rglob("data.yml")))
        candidates.extend(sorted(source_dir.rglob("dataset.yaml")))

        seen = set()
        for yaml_path in candidates:
            yaml_path = yaml_path.resolve()
            if yaml_path in seen:
                continue
            seen.add(yaml_path)
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
        default_master_id: Optional[int],
    ) -> Dict[int, Optional[int]]:
        """
        Build {native_class_id: master_class_id} mapping.

        If native_names is empty (no data.yaml), we try to match
        the raw integer IDs as string keys in the lookup.
        """
        remap = {}

        if native_names:
            for native_id, native_name in native_names.items():
                key = self.normalize_class_name(native_name)
                master_id = lookup.get(key, default_master_id)
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
        default_master_id: Optional[int],
        width: int,
        height: int,
    ) -> List[CanonicalObject]:
        """
        Read a YOLO label file and remap class IDs.
        Returns list of remapped lines (drops unmapped classes).
        """
        remapped: List[CanonicalObject] = []

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
                key = self.normalize_class_name(native_name)
                master_id = lookup.get(key, default_master_id)
                if master_id is not None and key not in lookup:
                    stats["auto_mapped_classes"].append(native_name)
                # Cache for future lookups
                id_remap[native_id] = master_id

            if master_id is None:
                # Unmapped class — track and skip
                native_name = native_names.get(native_id, str(native_id))
                stats["unmapped_classes"].append(native_name)
                continue

            inspected = self.inspect_yolo_bbox(parts, width, height)
            sanitized_bbox = inspected["sanitized_bbox"] or []

            stats["classes_found"][master_id] = (
                stats["classes_found"].get(master_id, 0) + 1
            )

            remapped.append(
                CanonicalObject(
                    bbox=sanitized_bbox,
                    category=master_id,
                    category_name="",
                    area=round(sanitized_bbox[2] * sanitized_bbox[3], 4) if sanitized_bbox else 0.0,
                    raw_bbox=inspected["raw_bbox"],
                    quality_flags=inspected["quality_flags"],
                    valid_geometry=bool(inspected["valid_geometry"]),
                )
            )

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
        for split_name in ("train", "val", "valid", "validation", "test"):
            img_dir = source_dir / "images" / split_name
            lbl_dir = source_dir / "labels" / split_name

            if img_dir.exists():
                actual_split = self._normalize_split_name(split_name)
                splits[actual_split] = (img_dir, lbl_dir if lbl_dir.exists() else None)

        if splits:
            return splits

        # Layout B: source_dir/{train,valid}/{images,labels}/
        for split_name in ("train", "val", "valid", "validation", "test"):
            split_dir = source_dir / split_name
            if split_dir.exists():
                img_dir = split_dir / "images"
                lbl_dir = split_dir / "labels"
                if img_dir.exists():
                    actual_split = self._normalize_split_name(split_name)
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

    def _discover_items_recursively(
        self,
        source_dir: Path,
        source_name: str,
    ) -> List[Tuple[Path, Optional[Path], str]]:
        """
        Fallback discovery for non-standard YOLO exports.

        Some Kaggle datasets ship with unusual nesting or capitalization
        (for example Train/Images, image files mixed into label folders, or
        annotations beside images). When our canonical layouts fail, scan the
        tree directly and recover image/label pairs heuristically.
        """
        label_index: DefaultDict[str, List[Path]] = defaultdict(list)
        for txt_path in source_dir.rglob("*.txt"):
            if txt_path.is_file():
                label_index[txt_path.stem.lower()].append(txt_path)

        all_items: List[Tuple[Path, Optional[Path], str]] = []
        for img_path in sorted(p for p in source_dir.rglob("*") if p.is_file() and self.is_image_file(p)):
            label_path = self._find_label_flexible(img_path, source_dir, label_index)
            split = self._infer_split_from_path(img_path, source_dir)
            all_items.append((img_path, label_path, split))

        if all_items:
            logger.info(
                f"[{source_name}] Recursive fallback discovered {len(all_items)} images"
            )
        else:
            logger.error(f"[{source_name}] Recursive fallback also found no images in {source_dir}")
        return all_items

    @staticmethod
    def _normalize_split_name(split_name: str) -> str:
        """
        Collapse source-specific split names into the standard YOLO layout.
        Final datasets should only contain train/val directories.
        """
        normalized = split_name.lower().strip()
        if normalized in {"val", "valid", "validation", "test"}:
            return "val"
        return "train"

    @staticmethod
    def _find_label(img_path: Path, lbl_dir: Optional[Path]) -> Optional[Path]:
        """Find the label file corresponding to an image."""
        if lbl_dir is None:
            # Try same directory as image
            lbl_path = img_path.with_suffix(".txt")
            return lbl_path if lbl_path.exists() else None

        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        return lbl_path if lbl_path.exists() else None

    def _find_label_flexible(
        self,
        img_path: Path,
        source_dir: Path,
        label_index: DefaultDict[str, List[Path]],
    ) -> Optional[Path]:
        """Best-effort label matching for non-standard YOLO directory layouts."""
        direct_candidates: List[Path] = [img_path.with_suffix(".txt")]

        try:
            rel_path = img_path.resolve().relative_to(source_dir.resolve())
        except ValueError:
            rel_path = Path(img_path.name)

        rel_parts = list(rel_path.parts)
        for i, part in enumerate(rel_parts[:-1]):
            if part.lower() == "images":
                candidate_parts = list(rel_parts)
                candidate_parts[i] = "labels"
                direct_candidates.append((source_dir / Path(*candidate_parts)).with_suffix(".txt"))

        for candidate in direct_candidates:
            if candidate.exists():
                return candidate

        stem_matches = label_index.get(img_path.stem.lower(), [])
        if len(stem_matches) == 1:
            return stem_matches[0]
        if len(stem_matches) > 1:
            return min(
                stem_matches,
                key=lambda candidate: self._label_distance_score(candidate, img_path, source_dir),
            )
        return None

    @staticmethod
    def _infer_split_from_path(img_path: Path, source_dir: Path) -> str:
        """Infer split from any parent folder name, defaulting to train."""
        try:
            parts = [part.lower() for part in img_path.resolve().relative_to(source_dir.resolve()).parts]
        except ValueError:
            parts = [part.lower() for part in img_path.parts]

        for part in parts:
            if part in {"val", "valid", "validation", "test"}:
                return "val"
            if part == "train":
                return "train"
        return "train"

    @staticmethod
    def _label_distance_score(label_path: Path, img_path: Path, source_dir: Path) -> Tuple[int, int, str]:
        """
        Prefer labels that live nearest to the image and share the most path
        structure after normalizing split/images/labels folder names.
        """
        normalized_tokens = {"train", "val", "valid", "validation", "test", "images", "labels"}

        def normalize(path: Path) -> List[str]:
            try:
                parts = path.resolve().relative_to(source_dir.resolve()).parts
            except ValueError:
                parts = path.parts
            return [part.lower() for part in parts[:-1] if part.lower() not in normalized_tokens]

        img_parts = normalize(img_path)
        label_parts = normalize(label_path)

        suffix_overlap = 0
        for img_part, label_part in zip(reversed(img_parts), reversed(label_parts)):
            if img_part != label_part:
                break
            suffix_overlap += 1

        try:
            distance = len(label_path.resolve().relative_to(source_dir.resolve()).parts)
        except ValueError:
            distance = len(label_path.parts)
        return (-suffix_overlap, distance, str(label_path))
