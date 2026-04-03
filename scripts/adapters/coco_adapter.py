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

from scripts.adapters.base import BaseAdapter, CanonicalObject, CanonicalSample

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
        default_master_id = self.infer_single_target_class(class_map)

        stats = {
            "images_processed": 0,
            "labels_written": 0,
            "classes_found": {},
            "unmapped_classes": [],
            "auto_mapped_classes": [],
            "samples": [],
            "errors": 0,
        }

        # 1. Find annotation JSON files
        json_files = self._discover_annotations(source_dir)
        if not json_files:
            logger.error(f"[{source_name}] No COCO JSON annotations found in {source_dir}")
            return stats

        # 2. Process each JSON file (there may be one per split)
        all_items: List[Tuple[Path, List[CanonicalObject], str, int, int]] = []

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
                key = self.normalize_class_name(cat_name)
                master_id = lookup.get(key, default_master_id)
                cat_remap[cat_id] = master_id
                if master_id is None:
                    stats["unmapped_classes"].append(cat_name)
                elif key not in lookup:
                    stats["auto_mapped_classes"].append(cat_name)

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

                # Convert annotations to canonical objects
                canonical_objects: List[CanonicalObject] = []
                for ann in anns_by_image.get(img_id, []):
                    cat_id = ann["category_id"]
                    master_id = cat_remap.get(cat_id)
                    if master_id is None:
                        continue

                    bbox = self.sanitize_coco_bbox(ann["bbox"], img_w, img_h)
                    if bbox is not None:
                        area = float(bbox[2] * bbox[3])
                        canonical_objects.append(
                            CanonicalObject(
                                bbox=[round(float(v), 4) for v in bbox],
                                category=master_id,
                                category_name="",
                                area=round(area, 4),
                            )
                        )
                        stats["classes_found"][master_id] = (
                            stats["classes_found"].get(master_id, 0) + 1
                        )

                if canonical_objects:
                    all_items.append(
                        (img_path, canonical_objects, inferred_split, img_w, img_h)
                    )

        # 3. Apply split strategy
        if split_strategy == "auto" or all(s == "train" for _, _, s, _, _ in all_items):
            random.seed(42)
            random.shuffle(all_items)
            n_val = max(1, int(len(all_items) * val_ratio))
            all_items = [
                (img, lines, "val" if i < n_val else "train", img_w, img_h)
                for i, (img, lines, _, img_w, img_h) in enumerate(all_items)
            ]
            logger.info(
                f"[{source_name}] Auto-split: "
                f"{len(all_items) - n_val} train / {n_val} val"
            )

        # 4. Write to staging
        for img_path, canonical_objects, split, img_w, img_h in all_items:
            try:
                new_name = self.build_output_name(source_name, img_path, source_dir)
                sample = CanonicalSample(
                    image_id=Path(new_name).stem,
                    image_path=str(img_path.resolve()),
                    output_file_name=new_name,
                    width=img_w,
                    height=img_h,
                    split=split,
                    objects=canonical_objects,
                    source_name=source_name,
                )
                stats["samples"].append(sample)

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
        if stats["auto_mapped_classes"]:
            logger.info(
                f"[{source_name}] Auto-mapped variant categories: "
                f"{list(set(stats['auto_mapped_classes']))}"
            )

        return stats

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _discover_annotations(
        self, source_dir: Path
    ) -> List[Tuple[Path, str]]:
        """
        Find COCO JSON files and infer their split.
        Returns [(json_path, split_name), ...].
        """
        results = []

        # Search recursively because many Kaggle archives nest the COCO files.
        for name_pattern in ("*.json",):
            for json_path in source_dir.rglob(name_pattern):
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
