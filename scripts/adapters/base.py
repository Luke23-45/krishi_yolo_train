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
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("krishi.adapter")


@dataclass
class CanonicalObject:
    """One canonical object annotation in COCO xywh pixel format."""

    bbox: List[float]
    category: int
    category_name: str
    area: float
    iscrowd: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CanonicalSample:
    """One canonical image record for the Hugging Face dataset."""

    image_id: str
    image_path: str
    output_file_name: str
    width: int
    height: int
    split: str
    objects: List[CanonicalObject]
    source_name: str
    source_type: str = ""
    source_handle: str = ""

    def to_metadata_row(self, sha256: str) -> Dict:
        return {
            "image_id": self.image_id,
            "file_name": self.output_file_name,
            "width": self.width,
            "height": self.height,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "source_handle": self.source_handle,
            "split": self.split,
            "objects": {
                "bbox": [obj.bbox for obj in self.objects],
                "category": [obj.category for obj in self.objects],
                "category_name": [obj.category_name for obj in self.objects],
                "area": [obj.area for obj in self.objects],
                "iscrowd": [obj.iscrowd for obj in self.objects],
            },
            "num_objects": len(self.objects),
            "sha256": sha256,
        }


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
            lookup[BaseAdapter.normalize_class_name(key_str)] = int(value)
        return lookup

    @staticmethod
    def normalize_class_name(name: str) -> str:
        """Normalize class labels from different tools into a stable key."""
        normalized = re.sub(r"[\W_]+", " ", str(name).lower()).strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    @staticmethod
    def infer_single_target_class(class_map: Dict) -> int | None:
        """
        If a source maps every native class to the same master class,
        unseen label variants can safely fall back to that target.
        """
        targets = {int(value) for value in class_map.values()}
        if len(targets) == 1:
            return next(iter(targets))
        return None

    @staticmethod
    def get_image_size(image_path: Path) -> tuple[int, int]:
        """Read image width and height."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required for canonical dataset generation. "
                "Install with: pip install Pillow"
            ) from exc

        with Image.open(image_path) as img:
            width, height = img.size
        return int(width), int(height)

    @staticmethod
    def yolo_bbox_to_coco(parts: List[str], width: int, height: int) -> Optional[List[float]]:
        """Convert YOLO normalized xywh values to COCO pixel xywh."""
        if len(parts) < 5:
            return None

        try:
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_w = float(parts[3])
            box_h = float(parts[4])
        except ValueError:
            return None

        x = (x_center - box_w / 2.0) * width
        y = (y_center - box_h / 2.0) * height
        w = box_w * width
        h = box_h * height

        x = max(0.0, min(float(width), x))
        y = max(0.0, min(float(height), y))
        w = max(0.0, min(float(width), w))
        h = max(0.0, min(float(height), h))

        if w <= 0.0 or h <= 0.0:
            return None

        return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]

    @staticmethod
    def sanitize_coco_bbox(
        bbox: List[float],
        width: int,
        height: int,
    ) -> Optional[List[float]]:
        """Clamp a COCO xywh pixel bbox to image bounds."""
        if len(bbox) != 4:
            return None

        try:
            x, y, w, h = [float(v) for v in bbox]
        except ValueError:
            return None

        if w <= 0.0 or h <= 0.0:
            return None

        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + w))
        y2 = max(0.0, min(float(height), y + h))

        new_w = x2 - x1
        new_h = y2 - y1
        if new_w <= 0.0 or new_h <= 0.0:
            return None

        return [round(x1, 4), round(y1, 4), round(new_w, 4), round(new_h, 4)]
