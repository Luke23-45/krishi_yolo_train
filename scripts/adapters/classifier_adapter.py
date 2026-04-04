"""
scripts/adapters/classifier_adapter.py
---------------------------------------
Adapter for image classification datasets (folder-per-class).

Input structure:
    source_dir/
    ├── ClassA/
    │   ├── img_001.jpg
    │   └── img_002.jpg
    ├── ClassB/
    │   └── img_003.jpg
    └── ClassC/
        └── img_004.jpg

This is a BOOTSTRAP adapter. It creates full-frame bounding boxes
(one box covering the entire image) for each classified image.
This is NOT ideal for detection training — it's a stopgap until
proper bounding-box annotated data is available.

The adapter:
    1. Scans for class-named subdirectories
    2. Maps folder names → master Krishi class IDs
    3. Creates YOLO labels with full-frame bbox: "<class_id> 0.5 0.5 1.0 1.0"
    4. Copies images with deduplicated names to staging

WARNING: Full-frame bounding boxes teach the model "the entire image IS
the object." This is acceptable for leaf detection (where images are
typically close-up leaf photos) but poor for localization accuracy.
Use this only when no proper detection dataset is available.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from scripts.adapters.base import BaseAdapter, CanonicalObject, CanonicalSample

logger = logging.getLogger("krishi.adapter.classifier")


class ClassifierAdapter(BaseAdapter):
    """Converts classification folder datasets to YOLO with full-frame bboxes."""

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
            "samples": [],
            "errors": 0,
        }

        # 1. Discover class folders
        # Look for subdirectories that contain images
        class_dirs = self._discover_class_dirs(source_dir)

        if not class_dirs:
            logger.error(
                f"[{source_name}] No class directories with images found in {source_dir}"
            )
            return stats

        logger.info(
            f"[{source_name}] Found {len(class_dirs)} class directories: "
            f"{[d.name for d in class_dirs]}"
        )

        # 2. Map folder names to master IDs
        all_items: List[Tuple[Path, int]] = []

        for class_dir in class_dirs:
            folder_name = class_dir.name
            key = folder_name.lower().strip()
            master_id = (
                lookup.get(key)
                or lookup.get(key.replace("_", " "))
                or lookup.get(key.replace(" ", "_"))
                or lookup.get(key.replace("-", "_"))
            )

            if master_id is None:
                stats["unmapped_classes"].append(folder_name)
                logger.warning(
                    f"[{source_name}] Folder '{folder_name}' not in class_map, skipping"
                )
                continue

            images = sorted(
                [f for f in class_dir.iterdir() if self.is_image_file(f)]
            )

            for img_path in images:
                all_items.append((img_path, master_id))

        if not all_items:
            logger.error(f"[{source_name}] No mappable images found")
            return stats

        # 3. Split
        random.seed(42)
        random.shuffle(all_items)
        n_val = max(1, int(len(all_items) * val_ratio))

        # 4. Process
        for i, (img_path, master_id) in enumerate(all_items):
            split = "val" if i < n_val else "train"

            try:
                new_name = self.build_output_name(source_name, img_path, source_dir)
                width, height = self.get_image_size(img_path)
                bbox = [0.0, 0.0, float(width), float(height)]
                sample = CanonicalSample(
                    image_id=Path(new_name).stem,
                    image_path=str(img_path.resolve()),
                    output_file_name=new_name,
                    width=width,
                    height=height,
                    split=split,
                    objects=[
                        CanonicalObject(
                            bbox=bbox,
                            category=master_id,
                            category_name="",
                            area=round(float(width) * float(height), 4),
                            raw_bbox=list(bbox),
                            quality_flags=[],
                            valid_geometry=True,
                        )
                    ],
                    source_name=source_name,
                )
                stats["samples"].append(sample)

                stats["images_processed"] += 1
                stats["labels_written"] += 1
                stats["classes_found"][master_id] = (
                    stats["classes_found"].get(master_id, 0) + 1
                )

            except Exception as e:
                logger.error(
                    f"[{source_name}] Error processing {img_path.name}: {e}"
                )
                stats["errors"] += 1

        logger.info(
            f"[{source_name}] Complete (BOOTSTRAP mode): "
            f"{stats['images_processed']} images, "
            f"{len(all_items) - n_val} train / {n_val} val, "
            f"{stats['errors']} errors"
        )

        return stats

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _discover_class_dirs(self, source_dir: Path) -> List[Path]:
        """
        Find directories that likely represent classes.
        Handles nested structures like source_dir/dataset/ClassA/
        """
        # Direct children
        candidates = [
            d for d in source_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        # Check if any candidate has images
        class_dirs = [
            d for d in candidates
            if any(self.is_image_file(f) for f in d.iterdir() if f.is_file())
        ]

        if class_dirs:
            return class_dirs

        # Try one level deeper (common: source_dir/dataset_name/ClassA/)
        for subdir in candidates:
            deeper = [
                d for d in subdir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
            deeper_with_images = [
                d for d in deeper
                if any(self.is_image_file(f) for f in d.iterdir() if f.is_file())
            ]
            if deeper_with_images:
                return deeper_with_images

        return []
