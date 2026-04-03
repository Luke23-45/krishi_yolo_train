"""
scripts/adapters/coco_adapter.py
---------------------------------
Adapter for datasets annotated in COCO JSON format.

COCO format uses a single JSON file containing:
    - "images": [{id, file_name, width, height}, ...]
    - "annotations": [{id, image_id, category_id, bbox: [x, y, w, h]}, ...]
    - "categories": [{id, name}, ...]

Where bbox is in absolute pixel coordinates [x_top_left, y_top_left, width, height].

This adapter:
    1. Discovers annotation JSON files in the source directory
    2. Parses COCO annotations and maps category names → master Krishi IDs
    3. Converts absolute pixel bboxes → YOLO normalized format
    4. Writes YOLO-format label files to staging
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scripts.adapters.base import BaseAdapter

logger = logging.getLogger("krishi.adapter.coco")


class COCOAdapter(BaseAdapter):
    """Converts COCO JSON annotations to YOLO detection format."""

    def process(
        self,
        source_dir: Path,
        staging_dir: Path,
        class_map: Dict[str, int],
        source_name: str,
        split_strategy: str = "auto",
        val_ratio: float = 0.2,
    ) -> Dict[str, int]:

        self.ensure_staging_dirs(staging_dir)
        lookup = self.build_class_map_lookup(class_map)

        stats = {
            "images_processed": 0,
            "labels_written": 0,
            "classes_found": {},
            "unmapped_classes": [],
            "errors": 0,
        }

        # 1. Find annotation JSON files
        json_files = self._discover_annotations(source_dir)
        if not json_files:
            logger.error(f"[{source_name}] No COCO JSON annotations found in {source_dir}")
            return stats

        # 2. Process each JSON file (there may be one per split)
        all_items: List[Tuple[Path, List[str], str]] = []

        for json_path, inferred_split in json_files:
            logger.info(f"[{source_name}] Processing {json_path.name} (split: {inferred_split})")

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    coco_data = json.load(f)
            except Exception as e:
                logger.error(f"[{source_name}] Failed to parse {json_path}: {e}")
                stats["errors"] += 1
                continue

            # Parse categories
            categories = {
                cat["id"]: cat["name"]
                for cat in coco_data.get("categories", [])
            }

            # Build category_id → master_id remap
            cat_remap = {}
            for cat_id, cat_name in categories.items():
                key = cat_name.lower().strip()
                master_id = (
                    lookup.get(key)
                    or lookup.get(key.replace("_", " "))
                    or lookup.get(key.replace(" ", "_"))
                )
                cat_remap[cat_id] = master_id
                if master_id is None:
                    stats["unmapped_classes"].append(cat_name)

            # Build image lookup
            images_meta = {
                img["id"]: img for img in coco_data.get("images", [])
            }

            # Group annotations by image
            anns_by_image: Dict[int, list] = defaultdict(list)
            for ann in coco_data.get("annotations", []):
                anns_by_image[ann["image_id"]].append(ann)

            # Resolve image directory
            img_dir = self._find_image_dir(source_dir, json_path)

            # Convert each image
            for img_id, img_meta in images_meta.items():
                img_file = img_meta["file_name"]
                img_w = img_meta.get("width", 0)
                img_h = img_meta.get("height", 0)

                if img_w <= 0 or img_h <= 0:
                    stats["errors"] += 1
                    continue

                img_path = img_dir / img_file
                if not img_path.exists():
                    # Try without subdirectory prefix in filename
                    img_path = img_dir / Path(img_file).name
                if not img_path.exists():
                    stats["errors"] += 1
                    continue

                # Convert annotations to YOLO lines
                yolo_lines = []
                for ann in anns_by_image.get(img_id, []):
                    cat_id = ann["category_id"]
                    master_id = cat_remap.get(cat_id)
                    if master_id is None:
                        continue

                    bbox = ann["bbox"]  # [x, y, w, h] in pixels
                    yolo_line = self._coco_bbox_to_yolo(
                        bbox, img_w, img_h, master_id
                    )
                    if yolo_line:
                        yolo_lines.append(yolo_line)
                        stats["classes_found"][master_id] = (
                            stats["classes_found"].get(master_id, 0) + 1
                        )

                if yolo_lines:
                    all_items.append((img_path, yolo_lines, inferred_split))

        # 3. Apply split strategy
        if split_strategy == "auto" or all(s == "train" for _, _, s in all_items):
            random.seed(42)
            random.shuffle(all_items)
            n_val = max(1, int(len(all_items) * val_ratio))
            all_items = [
                (img, lines, "val" if i < n_val else "train")
                for i, (img, lines, _) in enumerate(all_items)
            ]
            logger.info(
                f"[{source_name}] Auto-split: "
                f"{len(all_items) - n_val} train / {n_val} val"
            )

        # 4. Write to staging
        for img_path, yolo_lines, split in all_items:
            try:
                new_name = self.build_output_name(source_name, img_path, source_dir)
                label_name = Path(new_name).with_suffix(".txt").name

                self.copy_image(img_path, staging_dir, split, new_name)
                self.write_label(staging_dir, split, label_name, yolo_lines)

                stats["images_processed"] += 1
                stats["labels_written"] += 1
            except Exception as e:
                logger.error(f"[{source_name}] Error writing {img_path.name}: {e}")
                stats["errors"] += 1

        logger.info(
            f"[{source_name}] Complete: "
            f"{stats['images_processed']} images, "
            f"{stats['labels_written']} labels, "
            f"{stats['errors']} errors"
        )
        if stats["unmapped_classes"]:
            logger.warning(
                f"[{source_name}] Unmapped categories: "
                f"{list(set(stats['unmapped_classes']))}"
            )

        return stats

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    @staticmethod
    def _coco_bbox_to_yolo(
        bbox: list, img_w: int, img_h: int, class_id: int
    ) -> Optional[str]:
        """
        Convert COCO bbox [x_top_left, y_top_left, width, height] (pixels)
        to YOLO format: "<class_id> <x_center> <y_center> <width> <height>" (normalized).
        """
        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            return None

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        # Clamp to [0, 1]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

    def _discover_annotations(
        self, source_dir: Path
    ) -> List[Tuple[Path, str]]:
        """
        Find COCO JSON files and infer their split.
        Returns [(json_path, split_name), ...].
        """
        results = []

        # Check common locations
        for name_pattern in (
            "*.json",
            "annotations/*.json",
            "annotation/*.json",
        ):
            for json_path in source_dir.glob(name_pattern):
                if json_path.stat().st_size < 100:
                    continue  # Skip tiny files

                # Infer split from filename
                lower_name = json_path.stem.lower()
                if "val" in lower_name or "valid" in lower_name or "test" in lower_name:
                    split = "val"
                elif "train" in lower_name:
                    split = "train"
                else:
                    split = "train"  # Default to train

                results.append((json_path, split))

        # Deduplicate
        seen = set()
        unique = []
        for path, split in results:
            if path not in seen:
                seen.add(path)
                unique.append((path, split))

        return unique

    @staticmethod
    def _find_image_dir(source_dir: Path, json_path: Path) -> Path:
        """
        Determine where images are located relative to the annotation JSON.
        Tries several common conventions.
        """
        candidates = [
            source_dir / "images",
            source_dir / "train",
            source_dir / "train" / "images",
            json_path.parent / "images",
            json_path.parent,
            source_dir,
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # Check if it actually contains images
                has_images = any(
                    f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                    for f in candidate.iterdir()
                    if f.is_file()
                )
                if has_images:
                    return candidate

        # Fallback to source root
        return source_dir
