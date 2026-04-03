"""
scripts/validate.py
--------------------
Post-materialization validation for the Krishi Vaidya bouncer dataset.

Performs integrity checks:
    1. Image ↔ label file alignment (every image has a label, every label has an image)
    2. Label format integrity (valid class IDs, coordinates in [0, 1])
    3. Class distribution analysis with imbalance warnings
    4. Empty label detection
    5. Image readability check (optional, slower)

Usage:
    python scripts/validate.py --dataset krishi_bouncer_dataset
    python scripts/validate.py --dataset krishi_bouncer_dataset --check-images
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishi.validate")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================================
# VALIDATION CHECKS
# ============================================================================

def check_structure(dataset_dir: Path) -> bool:
    """Verify the directory structure is correct."""
    logger.info("─" * 50)
    logger.info("CHECK 1: Directory Structure")
    logger.info("─" * 50)

    required = [
        dataset_dir / "data.yaml",
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    ]

    all_ok = True
    for path in required:
        exists = path.exists()
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {path.relative_to(dataset_dir)}")
        if not exists:
            all_ok = False

    return all_ok


def check_data_yaml(dataset_dir: Path) -> Dict:
    """Validate data.yaml contents."""
    logger.info("─" * 50)
    logger.info("CHECK 2: data.yaml Schema")
    logger.info("─" * 50)

    data_yaml_path = dataset_dir / "data.yaml"
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    nc = cfg.get("nc", 0)
    names = cfg.get("names", {})

    logger.info(f"  Classes (nc): {nc}")
    for cls_id, cls_name in sorted(names.items()):
        logger.info(f"    {cls_id}: {cls_name}")

    if len(names) != nc:
        logger.error(f"  ✗ Mismatch: nc={nc} but {len(names)} names defined")
    else:
        logger.info(f"  ✓ Schema valid: {nc} classes")

    return cfg


def check_alignment(dataset_dir: Path) -> Dict[str, Dict]:
    """Check that every image has a label and vice versa."""
    logger.info("─" * 50)
    logger.info("CHECK 3: Image ↔ Label Alignment")
    logger.info("─" * 50)

    results = {}

    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split

        images: Set[str] = set()
        labels: Set[str] = set()

        if img_dir.exists():
            images = {
                f.stem for f in img_dir.iterdir()
                if f.suffix.lower() in IMAGE_EXTENSIONS
            }

        if lbl_dir.exists():
            labels = {
                f.stem for f in lbl_dir.iterdir()
                if f.suffix.lower() == ".txt"
            }

        orphan_images = images - labels  # Images without labels
        orphan_labels = labels - images  # Labels without images
        aligned = images & labels

        results[split] = {
            "images": len(images),
            "labels": len(labels),
            "aligned": len(aligned),
            "orphan_images": len(orphan_images),
            "orphan_labels": len(orphan_labels),
        }

        status = "✓" if (orphan_images == set() and orphan_labels == set()) else "⚠"
        logger.info(
            f"  {status} {split:5s} │ "
            f"{len(images):>6,d} images │ "
            f"{len(labels):>6,d} labels │ "
            f"{len(aligned):>6,d} aligned"
        )

        if orphan_images:
            logger.warning(
                f"    {len(orphan_images)} images have no label file"
            )
            for name in sorted(list(orphan_images))[:5]:
                logger.warning(f"      - {name}")

        if orphan_labels:
            logger.warning(
                f"    {len(orphan_labels)} labels have no image file"
            )
            for name in sorted(list(orphan_labels))[:5]:
                logger.warning(f"      - {name}")

    return results


def check_label_integrity(dataset_dir: Path, nc: int) -> Dict:
    """Validate all label files for format correctness."""
    logger.info("─" * 50)
    logger.info("CHECK 4: Label Format Integrity")
    logger.info("─" * 50)

    class_counter: Counter = Counter()
    total_boxes = 0
    errors: List[str] = []
    empty_labels = 0

    for split in ("train", "val"):
        lbl_dir = dataset_dir / "labels" / split
        if not lbl_dir.exists():
            continue

        for lbl_file in sorted(lbl_dir.iterdir()):
            if lbl_file.suffix.lower() != ".txt":
                continue

            content = lbl_file.read_text(encoding="utf-8").strip()
            if not content:
                empty_labels += 1
                continue

            for line_num, line in enumerate(content.splitlines(), 1):
                parts = line.strip().split()

                if len(parts) < 5:
                    errors.append(
                        f"{lbl_file.name}:{line_num} — "
                        f"Expected ≥5 values, got {len(parts)}"
                    )
                    continue

                try:
                    cls_id = int(parts[0])
                except ValueError:
                    errors.append(
                        f"{lbl_file.name}:{line_num} — "
                        f"Invalid class ID: '{parts[0]}'"
                    )
                    continue

                if cls_id < 0 or cls_id >= nc:
                    errors.append(
                        f"{lbl_file.name}:{line_num} — "
                        f"Class ID {cls_id} out of range [0, {nc - 1}]"
                    )

                # Validate coordinates
                try:
                    coords = [float(p) for p in parts[1:5]]
                    for j, val in enumerate(coords):
                        if val < -0.01 or val > 1.01:  # Small tolerance
                            errors.append(
                                f"{lbl_file.name}:{line_num} — "
                                f"Coordinate {j} out of range: {val:.4f}"
                            )
                except ValueError as e:
                    errors.append(
                        f"{lbl_file.name}:{line_num} — "
                        f"Invalid coordinate: {e}"
                    )
                    continue

                class_counter[cls_id] += 1
                total_boxes += 1

    logger.info(f"  Total bounding boxes: {total_boxes:,d}")
    logger.info(f"  Empty label files: {empty_labels}")
    logger.info(f"  Format errors: {len(errors)}")

    if errors:
        logger.warning("  First 10 errors:")
        for err in errors[:10]:
            logger.warning(f"    - {err}")

    return {
        "total_boxes": total_boxes,
        "empty_labels": empty_labels,
        "format_errors": len(errors),
        "class_counts": dict(class_counter),
    }


def print_class_distribution(class_counts: Dict[int, int], names: Dict) -> None:
    """Print a visual class distribution histogram."""
    logger.info("─" * 50)
    logger.info("CLASS DISTRIBUTION")
    logger.info("─" * 50)

    if not class_counts:
        logger.warning("  No annotations found!")
        return

    max_count = max(class_counts.values()) if class_counts else 1
    total = sum(class_counts.values())

    for cls_id in sorted(names.keys()):
        cls_name = names[cls_id]
        count = class_counts.get(cls_id, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar_len = int(count / max_count * 40) if max_count > 0 else 0
        bar = "█" * bar_len

        status = " "
        if count == 0:
            status = "✗"
        elif pct < 2.0:
            status = "⚠"

        logger.info(
            f"  {status} {cls_id:2d} │ {cls_name:20s} │ "
            f"{count:>7,d} ({pct:5.1f}%) │ {bar}"
        )

    # Imbalance warning
    if class_counts:
        min_count = min(class_counts.values())
        max_count_val = max(class_counts.values())
        ratio = max_count_val / min_count if min_count > 0 else float("inf")

        if ratio > 10:
            logger.warning(
                f"\n  ⚠ SEVERE CLASS IMBALANCE: "
                f"max/min ratio = {ratio:.0f}x"
            )
            logger.warning(
                "  Consider: Roboflow auto-augmentation, class weights, "
                "or oversampling underrepresented classes."
            )
        elif ratio > 5:
            logger.warning(
                f"\n  ⚠ Moderate class imbalance: "
                f"max/min ratio = {ratio:.0f}x"
            )


def check_images(dataset_dir: Path) -> Dict:
    """Optional: verify images are readable."""
    logger.info("─" * 50)
    logger.info("CHECK 5: Image Readability (slow)")
    logger.info("─" * 50)

    try:
        from PIL import Image
    except ImportError:
        logger.warning("  Pillow not installed, skipping image checks")
        return {"skipped": True}

    corrupt = []
    sizes: Counter = Counter()
    total = 0

    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        if not img_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            total += 1

            try:
                with Image.open(img_path) as img:
                    img.verify()
                    sizes[f"{img.size[0]}x{img.size[1]}"] += 1
            except Exception as e:
                corrupt.append(f"{img_path.name}: {e}")

    logger.info(f"  Checked: {total:,d} images")
    logger.info(f"  Corrupt: {len(corrupt)}")

    if corrupt:
        for c in corrupt[:5]:
            logger.warning(f"    - {c}")

    if sizes:
        logger.info("  Image sizes:")
        for size, count in sizes.most_common(5):
            logger.info(f"    {size}: {count:,d}")

    return {
        "total_checked": total,
        "corrupt": len(corrupt),
        "sizes": dict(sizes.most_common(10)),
    }


# ============================================================================
# MAIN
# ============================================================================

def validate(dataset_dir: Path, do_check_images: bool = False) -> None:
    """Run all validation checks."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║   KRISHI VAIDYA — Bouncer Dataset Validator v1.0        ║
    ╚══════════════════════════════════════════════════════════╝
    """
    logger.info(banner)
    logger.info(f"  Dataset: {dataset_dir.resolve()}")

    if not dataset_dir.exists():
        logger.error(f"Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    # Run checks
    structure_ok = check_structure(dataset_dir)
    if not structure_ok:
        logger.error("Structure check failed. Cannot continue.")
        sys.exit(1)

    cfg = check_data_yaml(dataset_dir)
    nc = cfg.get("nc", 10)
    names = cfg.get("names", {})

    alignment = check_alignment(dataset_dir)
    integrity = check_label_integrity(dataset_dir, nc)
    print_class_distribution(integrity.get("class_counts", {}), names)

    image_results = {}
    if do_check_images:
        image_results = check_images(dataset_dir)

    # Summary
    logger.info("═" * 50)
    logger.info("VALIDATION SUMMARY")
    logger.info("═" * 50)

    total_images = sum(
        a["images"] for a in alignment.values()
    )
    total_aligned = sum(
        a["aligned"] for a in alignment.values()
    )
    total_boxes = integrity.get("total_boxes", 0)
    total_errors = integrity.get("format_errors", 0)

    logger.info(f"  Total images:      {total_images:>8,d}")
    logger.info(f"  Aligned pairs:     {total_aligned:>8,d}")
    logger.info(f"  Total annotations: {total_boxes:>8,d}")
    logger.info(f"  Format errors:     {total_errors:>8,d}")

    if total_errors == 0 and total_images == total_aligned and total_boxes > 0:
        logger.info("")
        logger.info("  ✓ DATASET IS VALID — Ready for training.")
        logger.info("")
        logger.info("  Train with:")
        logger.info(f"    yolo detect train data={dataset_dir / 'data.yaml'} model=yolov8n.pt epochs=50 imgsz=640")
    else:
        logger.warning("")
        logger.warning("  ⚠ ISSUES DETECTED — Review the warnings above before training.")

    # Save validation report
    report = {
        "dataset": str(dataset_dir.resolve()),
        "structure_ok": structure_ok,
        "alignment": alignment,
        "integrity": integrity,
        "image_checks": image_results,
    }

    report_path = dataset_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"  Report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate a Krishi Vaidya bouncer dataset"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "krishi_bouncer_dataset",
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also verify image file readability (slower)",
    )

    args = parser.parse_args()
    validate(args.dataset, do_check_images=args.check_images)


if __name__ == "__main__":
    main()
