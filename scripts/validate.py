"""
Validation for canonical Hugging Face and derived YOLO datasets.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.canonical_dataset import IMAGE_EXTENSIONS, read_schema_names, read_split_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishi.validate")

YOLO_EPSILON = 1e-6


def resolve_dataset_path(path_value: str | Path) -> Path:
    expanded = Path(str(path_value)).expanduser()
    if expanded.is_absolute():
        return expanded
    return (PROJECT_ROOT / expanded).resolve()


def _load_curation_summary(dataset_dir: Path) -> Dict:
    report_path = dataset_dir / "materialization_report.json"
    if not report_path.exists():
        return {}
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload.get("curation", {}) or {}


def _check_images_readable(image_paths: List[Path]) -> Dict:
    try:
        from PIL import Image
    except ImportError:
        return {"skipped": True}

    corrupt = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as exc:
            corrupt.append(f"{image_path.name}: {exc}")
    return {"checked": len(image_paths), "corrupt": corrupt}


def validate_canonical(dataset_dir: Path, do_check_images: bool = False) -> None:
    logger.info("Validating canonical dataset: %s", dataset_dir)

    required = [
        dataset_dir / "classes.json",
        dataset_dir / "licenses.json",
        dataset_dir / "README.md",
        dataset_dir / "train" / "images",
        dataset_dir / "train" / "metadata.jsonl",
        dataset_dir / "val" / "images",
        dataset_dir / "val" / "metadata.jsonl",
        dataset_dir / "parquet" / "train_metadata.parquet",
        dataset_dir / "parquet" / "val_metadata.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        logger.error("Missing required canonical paths: %s", missing)
        sys.exit(1)

    names = read_schema_names(dataset_dir)
    class_counts: Counter = Counter()
    split_summary: Dict[str, Dict[str, int]] = {}
    errors: List[str] = []
    image_paths: List[Path] = []

    for split in ("train", "val"):
        rows = read_split_metadata(dataset_dir, split)
        split_dir = dataset_dir / split
        image_dir = split_dir / "images"
        files_on_disk = {
            path.name for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        rows_seen: Set[str] = set()

        for row in rows:
            file_name = row["file_name"]
            rows_seen.add(Path(file_name).name)
            image_path = split_dir / file_name
            image_paths.append(image_path)

            if not image_path.exists():
                errors.append(f"{split}/{file_name}: image referenced in parquet does not exist")
                continue

            objects = row["objects"]
            num_objects = int(row["num_objects"])
            if num_objects != len(objects["bbox"]):
                errors.append(f"{split}/{file_name}: num_objects does not match bbox count")
            expected_len = len(objects["bbox"])
            for key in ("categories", "category_names", "area", "iscrowd"):
                if len(objects[key]) != expected_len:
                    errors.append(f"{split}/{file_name}: objects.{key} length does not match bbox count")

            width = int(row["width"])
            height = int(row["height"])
            for category, bbox, area in zip(objects["categories"], objects["bbox"], objects["area"]):
                if int(category) not in names:
                    errors.append(f"{split}/{file_name}: unknown category id {category}")
                x, y, w, h = [float(v) for v in bbox]
                if w <= 0 or h <= 0:
                    errors.append(f"{split}/{file_name}: non-positive bbox size {bbox}")
                if x < 0 or y < 0 or x + w > width + 1e-3 or y + h > height + 1e-3:
                    errors.append(f"{split}/{file_name}: bbox out of bounds {bbox}")
                if float(area) <= 0:
                    errors.append(f"{split}/{file_name}: non-positive area {area}")
                class_counts[int(category)] += 1

        orphan_files = sorted(files_on_disk - rows_seen)
        if orphan_files:
            errors.extend([f"{split}/{name}: image exists on disk but missing from parquet" for name in orphan_files[:20]])

        split_summary[split] = {"images": len(rows), "files": len(files_on_disk)}

    image_results = _check_images_readable(image_paths) if do_check_images else {}
    curation_summary = _load_curation_summary(dataset_dir)

    logger.info("Canonical split counts: %s", split_summary)
    for class_id in sorted(names):
        logger.info("  %2d | %-20s | %7d", class_id, names[class_id], class_counts.get(class_id, 0))

    if errors:
        logger.warning("Found %s canonical validation issues", len(errors))
        for err in errors[:20]:
            logger.warning("  - %s", err)
    else:
        logger.info("Canonical dataset is valid.")
        if curation_summary:
            logger.info(
                "Curation summary: dropped %s images and %s objects during build.",
                curation_summary.get("dropped_images", 0),
                curation_summary.get("dropped_objects", 0),
            )

    report = {
        "format": "canonical",
        "dataset": str(dataset_dir),
        "split_summary": split_summary,
        "class_counts": dict(class_counts),
        "errors": errors,
        "image_checks": image_results,
        "curation": curation_summary,
    }
    report_path = dataset_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report saved: %s", report_path)
    if errors:
        sys.exit(1)


def validate_yolo(dataset_dir: Path, do_check_images: bool = False) -> None:
    logger.info("Validating YOLO dataset: %s", dataset_dir)

    required = [
        dataset_dir / "data.yaml",
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        logger.error("Missing required YOLO paths: %s", missing)
        sys.exit(1)

    with open(dataset_dir / "data.yaml", "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    nc = int(cfg["nc"])
    if isinstance(cfg["names"], list):
        names = {idx: name for idx, name in enumerate(cfg["names"])}
    else:
        names = {int(k): v for k, v in cfg["names"].items()}

    errors: List[str] = []
    class_counts: Counter = Counter()
    image_paths: List[Path] = []
    split_summary: Dict[str, Dict[str, int]] = {}

    for split in ("train", "val"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        image_stems = {
            path.stem for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        label_stems = {
            path.stem for path in label_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".txt"
        }
        if image_stems != label_stems:
            for name in sorted(image_stems - label_stems)[:20]:
                errors.append(f"{split}/{name}: image missing label")
            for name in sorted(label_stems - image_stems)[:20]:
                errors.append(f"{split}/{name}: label missing image")

        split_summary[split] = {"images": len(image_stems), "labels": len(label_stems)}

        for image_path in image_dir.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(image_path)

        for label_path in label_dir.iterdir():
            if not label_path.is_file() or label_path.suffix.lower() != ".txt":
                continue
            content = label_path.read_text(encoding="utf-8").strip()
            if not content:
                errors.append(f"{split}/{label_path.name}: empty label file")
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{split}/{label_path.name}:{line_no}: expected 5 fields")
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{split}/{label_path.name}:{line_no}: non-numeric values")
                    continue
                if cls_id < 0 or cls_id >= nc:
                    errors.append(f"{split}/{label_path.name}:{line_no}: class id {cls_id} out of range")
                if any(value < -YOLO_EPSILON or value > 1.0 + YOLO_EPSILON for value in coords):
                    errors.append(f"{split}/{label_path.name}:{line_no}: coordinate outside [0,1]")
                x_center, y_center, box_w, box_h = coords
                if (x_center - box_w / 2.0) < -YOLO_EPSILON or (x_center + box_w / 2.0) > 1.0 + YOLO_EPSILON:
                    errors.append(f"{split}/{label_path.name}:{line_no}: x bbox extends outside image")
                if (y_center - box_h / 2.0) < -YOLO_EPSILON or (y_center + box_h / 2.0) > 1.0 + YOLO_EPSILON:
                    errors.append(f"{split}/{label_path.name}:{line_no}: y bbox extends outside image")
                class_counts[cls_id] += 1

    image_results = _check_images_readable(image_paths) if do_check_images else {}
    curation_summary = _load_curation_summary(dataset_dir.parent / "hf_dataset") if not (dataset_dir / "materialization_report.json").exists() else _load_curation_summary(dataset_dir)

    logger.info("YOLO split counts: %s", split_summary)
    for class_id in sorted(names):
        logger.info("  %2d | %-20s | %7d", class_id, names[class_id], class_counts.get(class_id, 0))

    if errors:
        logger.warning("Found %s YOLO validation issues", len(errors))
        for err in errors[:20]:
            logger.warning("  - %s", err)
    else:
        logger.info("YOLO dataset is valid.")
        if curation_summary:
            logger.info(
                "Curation summary: dropped %s images and %s objects during build.",
                curation_summary.get("dropped_images", 0),
                curation_summary.get("dropped_objects", 0),
            )

    report = {
        "format": "yolo",
        "dataset": str(dataset_dir),
        "split_summary": split_summary,
        "class_counts": dict(class_counts),
        "errors": errors,
        "image_checks": image_results,
        "curation": curation_summary,
    }
    report_path = dataset_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Report saved: %s", report_path)
    if errors:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate canonical or YOLO dataset outputs")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset root path")
    parser.add_argument(
        "--format",
        choices=("canonical", "yolo"),
        default="yolo",
        help="Dataset format to validate",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also verify that image files are readable",
    )
    args = parser.parse_args()

    dataset_dir = resolve_dataset_path(args.dataset)
    if args.format == "canonical":
        validate_canonical(dataset_dir, do_check_images=args.check_images)
    else:
        validate_yolo(dataset_dir, do_check_images=args.check_images)


if __name__ == "__main__":
    main()
