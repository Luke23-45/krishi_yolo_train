"""
scripts/adapters/base.py
-------------------------
Abstract base class for dataset format adapters.

Every adapter must implement a single contract:
    process(source_dir, staging_dir, class_map, source_name, split_strategy, val_ratio)

The adapter reads from source_dir, remaps classes via class_map,
and writes YOLO-format images + labels into staging_dir.
"""

from __future__ import annotations

import abc
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("krishi.adapter")


class BaseAdapter(abc.ABC):
    """
    Contract that every format adapter must fulfill.

    After processing, the staging directory must contain:
        staging_dir/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/

    All label files must be in YOLO normalized format:
        <class_id> <x_center> <y_center> <width> <height>
    where class_id is from the master Krishi schema (0-11).
    """

    @abc.abstractmethod
    def process(
        self,
        source_dir: Path,
        staging_dir: Path,
        class_map: Dict[str, int],
        source_name: str,
        split_strategy: str = "preserve",
        val_ratio: float = 0.2,
    ) -> Dict[str, int]:
        """
        Process a single data source into staging.

        Args:
            source_dir:     Root of the downloaded/local dataset.
            staging_dir:    Where to write processed images + labels.
            class_map:      Maps source class names -> master Krishi IDs (0-11).
            source_name:    Unique prefix for filename deduplication.
            split_strategy: "preserve" to keep source splits, "auto" for random 80/20.
            val_ratio:      Fraction for validation when split_strategy is "auto".

        Returns:
            Dict with processing statistics:
                {"images_processed": int, "labels_written": int,
                 "classes_found": dict, "unmapped_classes": list,
                 "errors": int}
        """
        ...

    @staticmethod
    def ensure_staging_dirs(staging_dir: Path) -> None:
        """Create the canonical staging directory structure."""
        for split in ("train", "val"):
            (staging_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (staging_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def copy_image(
        src_path: Path,
        staging_dir: Path,
        split: str,
        new_name: str,
    ) -> Path:
        """Copy an image to staging with a deduplicated filename."""
        dst = staging_dir / "images" / split / new_name
        shutil.copy2(src_path, dst)
        return dst

    @staticmethod
    def write_label(
        staging_dir: Path,
        split: str,
        label_name: str,
        lines: List[str],
    ) -> Path:
        """Write a YOLO label file to staging."""
        dst = staging_dir / "labels" / split / label_name
        dst.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
        return dst

    @staticmethod
    def deduplicate_name(source_name: str, original_name: str) -> str:
        """Backward-compatible basename prefixing helper."""
        return f"{source_name}__{original_name}"

    @staticmethod
    def build_output_name(
        source_name: str,
        img_path: Path,
        source_dir: Path,
    ) -> str:
        """
        Build a deterministic output filename that stays unique even when a
        source reuses the same basename in multiple folders.
        """
        try:
            relative_path = img_path.resolve().relative_to(source_dir.resolve())
        except ValueError:
            relative_path = Path(img_path.name)

        safe_parts = []
        for part in relative_path.parent.parts:
            if part in ("", "."):
                continue
            sanitized = "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_"
                for ch in part
            ).strip("_")
            if sanitized:
                safe_parts.append(sanitized)

        stem = relative_path.stem
        suffix = img_path.suffix.lower()
        fingerprint = hashlib.sha1(
            str(relative_path).encode("utf-8")
        ).hexdigest()[:10]

        if safe_parts:
            return (
                f"{source_name}__{'__'.join(safe_parts)}__"
                f"{stem}__{fingerprint}{suffix}"
            )
        return f"{source_name}__{stem}__{fingerprint}{suffix}"

    @staticmethod
    def is_image_file(path: Path) -> bool:
        """Check if a path is a supported image format."""
        return path.suffix.lower() in {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

    @staticmethod
    def build_class_map_lookup(class_map: Dict) -> Dict[str, int]:
        """
        Build a case-insensitive lookup from class_map.
        Allows matching 'Panicle', 'panicle', 'PANICLE' etc.

        Handles YAML auto-parsing of keys as integers (e.g., class IDs)
        by coercing everything to lowercase strings.
        """
        lookup = {}
        for key, value in class_map.items():
            key_str = str(key).lower().strip()
            lookup[key_str] = int(value)
            lookup[key_str.replace("_", " ")] = int(value)
            lookup[key_str.replace(" ", "_")] = int(value)
            lookup[key_str.replace("-", "_")] = int(value)
        return lookup
