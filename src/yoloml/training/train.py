"""
yoloml/training/train.py
--------------------------
End-to-end YOLOv8 training pipeline with class-imbalance mitigation.

Implements two complementary, research-backed strategies:

1. Offline Repeat Factor Sampling (RFS)
   Gupta et al., "LVIS: A Dataset for Large Vocabulary Instance Segmentation", CVPR 2019
   Creates a physically rebalanced copy of the training set by repeating images
   that contain rare classes. Avoids hacking the Ultralytics dataloader.

2. Class-Balanced Loss Weighting
   Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
   Computes per-class weights via the effective-number formula and injects them
   into the model's classification loss before training begins.

Usage:
    python -m yoloml.training.train \\
        --data krishi_bouncer_dataset/data.yaml \\
        --balance \\
        --epochs 100 \\
        --batch 16 \\
        --imgsz 640

    python -m yoloml.training.train --data krishi_bouncer_dataset/data.yaml --dry-run
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml
from yoloml.config import TelemetryConfig, TrainingConfig
from yoloml.pipeline import (
    DataManifest,
    TrainManifest,
    create_run_context,
    ensure_stage_dir,
    load_cli_config,
    parse_stage_args,
    read_manifest,
    snapshot_config,
    write_manifest,
)
from yoloml.utils.telemetry import setup_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishi.train")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def scan_class_distribution(label_dir: Path) -> Tuple[Counter, Dict[str, Set[int]]]:
    """
    Scan YOLO label files and return:
      - class_counts: Counter mapping class_id → total instance count
      - image_classes: dict mapping image_stem → set of class_ids present
    """
    class_counts: Counter = Counter()
    image_classes: Dict[str, Set[int]] = {}

    for label_path in sorted(label_dir.iterdir()):
        if not label_path.is_file() or label_path.suffix.lower() != ".txt":
            continue
        stem = label_path.stem
        classes_in_image: Set[int] = set()
        content = label_path.read_text(encoding="utf-8").strip()
        if not content:
            image_classes[stem] = classes_in_image
            continue
        for line in content.splitlines():
            parts = line.split()
            if len(parts) >= 5:
                cid = int(parts[0])
                class_counts[cid] += 1
                classes_in_image.add(cid)
        image_classes[stem] = classes_in_image

    return class_counts, image_classes


# ═══════════════════════════════════════════════════════════════════════════════
# REPEAT FACTOR SAMPLING (Gupta et al., CVPR 2019)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_repeat_factors(
    image_classes: Dict[str, Set[int]],
    threshold: Optional[float] = None,
) -> Dict[str, int]:
    """
    Compute per-image integer repeat factors using LVIS-style RFS.

    For each class c, the image-level frequency is:
        f(c) = |{images containing c}| / |{all images}|

    The per-class repeat factor is:
        r(c) = max(1, sqrt(t / f(c)))

    The per-image repeat factor is:
        R(i) = ceil(max(r(c) for c in classes_in_image_i))

    Args:
        image_classes: dict mapping image_stem → set of class_ids
        threshold: RFS threshold t. If None, auto-computed as the median
                   class frequency (robust default for long-tail distributions).

    Returns:
        dict mapping image_stem → integer repeat count (≥ 1)
    """
    total_images = len(image_classes)
    if total_images == 0:
        return {}

    class_image_counts: Counter = Counter()
    for classes in image_classes.values():
        for c in classes:
            class_image_counts[c] += 1

    freqs = {c: n / total_images for c, n in class_image_counts.items()}

    if threshold is None:
        sorted_freqs = sorted(freqs.values())
        threshold = sorted_freqs[len(sorted_freqs) // 2] if sorted_freqs else 0.01
        logger.info("RFS auto-threshold (median frequency): %.6f", threshold)

    class_factors = {}
    for c, f in freqs.items():
        class_factors[c] = max(1.0, math.sqrt(threshold / max(f, 1e-12)))

    repeat_factors: Dict[str, int] = {}
    for stem, classes in image_classes.items():
        if not classes:
            repeat_factors[stem] = 1
            continue
        max_factor = max(class_factors.get(c, 1.0) for c in classes)
        repeat_factors[stem] = math.ceil(max_factor)

    return repeat_factors


def create_balanced_dataset(
    source_dir: Path,
    output_dir: Path,
    repeat_factors: Dict[str, int],
) -> Dict[str, int]:
    """
    Create an RFS-balanced copy of the YOLO training set by physically
    duplicating images and labels for under-represented classes.

    Uses os.link() (hard links) when possible to avoid wasting disk space.
    Falls back to shutil.copy2() on cross-device or unsupported filesystems.

    Returns:
        stats dict with original_images, balanced_images, duplicated_images
    """
    src_images = source_dir / "images" / "train"
    src_labels = source_dir / "labels" / "train"
    dst_images = output_dir / "images" / "train"
    dst_labels = output_dir / "labels" / "train"

    for d in (dst_images, dst_labels):
        d.mkdir(parents=True, exist_ok=True)

    for split in ("val",):
        for sub in ("images", "labels"):
            src = source_dir / sub / split
            dst = output_dir / sub / split
            if src.exists():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            os.link(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)

    duplicated = 0
    total = 0

    for stem, factor in repeat_factors.items():
        src_label = src_labels / f"{stem}.txt"
        if not src_label.exists():
            continue

        src_image = None
        for ext in IMAGE_EXTENSIONS:
            candidate = src_images / f"{stem}{ext}"
            if candidate.exists():
                src_image = candidate
                break
        if src_image is None:
            continue

        _link_or_copy(src_image, dst_images / src_image.name)
        _link_or_copy(src_label, dst_labels / src_label.name)
        total += 1

        for k in range(1, factor):
            dup_stem = f"{stem}_rfs{k}"
            dup_image = dst_images / f"{dup_stem}{src_image.suffix}"
            dup_label = dst_labels / f"{dup_stem}.txt"
            _link_or_copy(src_image, dup_image)
            _link_or_copy(src_label, dup_label)
            duplicated += 1
            total += 1

    src_data_yaml = source_dir / "data.yaml"
    if src_data_yaml.exists():
        with open(src_data_yaml, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        cfg["path"] = str(output_dir.resolve())
        with open(output_dir / "data.yaml", "w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

    return {
        "original_images": len(repeat_factors),
        "balanced_images": total,
        "duplicated_images": duplicated,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLASS-BALANCED LOSS WEIGHTS (Cui et al., CVPR 2019)
# ═══════════════════════════════════════════════════════════════════════════════

def effective_number(n: float, beta: float) -> float:
    """Compute the effective number of samples: E_n = (1 - β^n) / (1 - β)."""
    if abs(beta - 1.0) < 1e-12:
        return float(n)
    if beta < 1e-12:
        return 1.0
    return (1.0 - beta ** n) / (1.0 - beta)


def compute_class_weights(
    class_counts: Dict[int, int],
    num_classes: int,
    beta: float = 0.9999,
) -> List[float]:
    """
    Compute normalized per-class loss weights using the effective number formula.

    Returns a list of length num_classes where weights[i] is the weight for class i.
    Classes with fewer samples receive higher weights.
    """
    eff_nums = []
    for c in range(num_classes):
        n = class_counts.get(c, 1)
        eff_nums.append(effective_number(max(n, 1), beta))

    inv_eff = [1.0 / e for e in eff_nums]
    total = sum(inv_eff)
    weights = [(w / total) * num_classes for w in inv_eff]
    return weights


def inject_class_weights(model, weights: List[float]) -> bool:
    """
    Inject per-class weights into the model's classification BCE loss.

    This patches the Detect head's BCEWithLogitsLoss with a weighted version.
    Falls back gracefully if the Ultralytics internal API has changed.
    """
    try:
        import torch

        det_head = model.model.model[-1]
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

        if hasattr(det_head, "bce"):
            det_head.bce = torch.nn.BCEWithLogitsLoss(
                pos_weight=weight_tensor,
                reduction="none",
            )
            logger.info("Injected class-balanced weights into detection loss (Cui et al.).")
            return True

        logger.warning(
            "Detection head does not expose 'bce' attribute. "
            "Class-balanced loss weighting skipped. RFS alone will handle imbalance."
        )
    except Exception as exc:
        logger.warning(
            "Could not inject class-balanced weights — Ultralytics API may have changed: %s. "
            "RFS alone will handle imbalance.",
            exc,
        )
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_device(requested: str) -> str:
    """Resolve the training device with CPU fallback."""
    if requested == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "0"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        logger.info("No GPU detected. Falling back to CPU.")
        return "cpu"
    return requested


def train(
    args: TrainingConfig,
    data_yaml: Path,
    telemetry_cfg: TelemetryConfig,
    output_root: Path,
    training_config_snapshot: Path,
    run_id: str,
    data_manifest_path: Optional[Path] = None,
) -> TrainManifest:
    if not data_yaml.exists():
        logger.error("data.yaml not found: %s", data_yaml)
        sys.exit(1)

    dataset_dir = data_yaml.parent

    with open(data_yaml, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    num_classes = int(cfg["nc"])
    if isinstance(cfg["names"], list):
        names = {i: n for i, n in enumerate(cfg["names"])}
    else:
        names = {int(k): v for k, v in cfg["names"].items()}

    logger.info("=" * 64)
    logger.info("  KRISHI VAIDYA — Training Pipeline v1.0")
    logger.info("=" * 64)
    logger.info("Dataset:     %s", dataset_dir)
    logger.info("Classes:     %d", num_classes)
    logger.info("Model:       %s", args.model)
    logger.info("Epochs:      %d", args.epochs)
    logger.info("Batch:       %d", args.batch)
    logger.info("Image size:  %d", args.imgsz)
    logger.info("Balance:     %s", "RFS enabled" if args.balance else "disabled")
    logger.info("Beta:        %s", args.beta)

    label_dir = dataset_dir / "labels" / "train"
    if not label_dir.exists():
        logger.error("Training labels not found: %s", label_dir)
        sys.exit(1)

    logger.info("-" * 64)
    logger.info("STEP 1: Scanning class distribution")
    logger.info("-" * 64)
    class_counts, image_classes = scan_class_distribution(label_dir)
    for cid in sorted(names.keys()):
        logger.info("  %2d | %-20s | %7d instances", cid, names[cid], class_counts.get(cid, 0))

    training_data_yaml = data_yaml

    if args.balance:
        logger.info("-" * 64)
        logger.info("STEP 2: Offline Repeat Factor Sampling (Gupta et al., CVPR 2019)")
        logger.info("-" * 64)

        repeat_factors = compute_repeat_factors(
            image_classes, threshold=args.rfs_threshold,
        )
        factor_dist = Counter(repeat_factors.values())
        for factor, count in sorted(factor_dist.items()):
            logger.info("  Repeat factor %d: %d images", factor, count)

        balanced_dir = dataset_dir.parent / f"{dataset_dir.name}_balanced"
        if balanced_dir.exists():
            shutil.rmtree(balanced_dir)

        logger.info("Creating balanced dataset at: %s", balanced_dir)
        stats = create_balanced_dataset(dataset_dir, balanced_dir, repeat_factors)
        logger.info(
            "  Original: %d images → Balanced: %d images (+%d duplicates)",
            stats["original_images"], stats["balanced_images"], stats["duplicated_images"],
        )

        balanced_counts, _ = scan_class_distribution(balanced_dir / "labels" / "train")
        logger.info("  Post-RFS distribution:")
        for cid in sorted(names.keys()):
            orig = class_counts.get(cid, 0)
            balanced = balanced_counts.get(cid, 0)
            ratio = balanced / max(orig, 1)
            logger.info(
                "    %2d | %-20s | %7d → %7d (×%.1f)",
                cid, names[cid], orig, balanced, ratio,
            )

        training_data_yaml = balanced_dir / "data.yaml"

    logger.info("-" * 64)
    logger.info("STEP 3: Computing class-balanced loss weights (Cui et al., CVPR 2019)")
    logger.info("-" * 64)
    weights = compute_class_weights(class_counts, num_classes, beta=args.beta)
    for cid in sorted(names.keys()):
        logger.info("  %2d | %-20s | weight = %.4f", cid, names[cid], weights[cid])

    balance_report = {
        "original_distribution": dict(class_counts),
        "class_weights": {names[c]: round(w, 6) for c, w in enumerate(weights)},
        "beta": args.beta,
        "rfs_enabled": args.balance,
    }
    if args.balance:
        balance_report["rfs_threshold"] = args.rfs_threshold or "auto"
        balance_report["repeat_factor_distribution"] = dict(Counter(repeat_factors.values()))

    report_dir = ensure_stage_dir(output_root)
    report_path = report_dir / "balance_report.json"
    report_path.write_text(json.dumps(balance_report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Balance report saved: %s", report_path)

    telemetry_run_name = telemetry_cfg.run_name or args.name or output_root.name
    manifest = TrainManifest(
        run_id=run_id,
        data_manifest=str(data_manifest_path) if data_manifest_path else None,
        verified_data_yaml=str(training_data_yaml.resolve()),
        training_config_snapshot=str(training_config_snapshot.resolve()),
        output_root=str(output_root.resolve()),
        balance_report=str(report_path.resolve()),
        telemetry_project=telemetry_cfg.project,
        telemetry_run_name=telemetry_run_name,
        best_weights=None,
        last_weights=None,
        valid=False,
    )

    if args.dry_run:
        logger.info("=" * 64)
        logger.info("DRY RUN complete. No training launched.")
        logger.info("=" * 64)
        manifest.valid = True
        return manifest

    logger.info("-" * 64)
    logger.info("STEP 4: Launching YOLOv8 training")
    logger.info("-" * 64)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)

    if not args.no_class_weights:
        inject_class_weights(model, weights)

    device = resolve_device(args.device)
    experiment_name = telemetry_run_name
    artifacts_dir = output_root / "artifacts"

    logger.info("Device:      %s", device)
    logger.info("Experiment:  %s", experiment_name)
    logger.info("Data YAML:   %s", training_data_yaml)
    logger.info("Artifacts:   %s", artifacts_dir)

    kwargs = {
        "data": str(training_data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(output_root),
        "name": "artifacts",
        "device": device,
        "patience": args.patience,
        "exist_ok": True,
    }

    results = model.train(**kwargs)

    save_dir = Path(getattr(results, "save_dir", getattr(model, "trainer", object()).save_dir if hasattr(getattr(model, "trainer", None), "save_dir") else artifacts_dir))
    if not save_dir.is_absolute():
        save_dir = (PROJECT_ROOT / save_dir).resolve()
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    manifest.best_weights = str(best_weights.resolve()) if best_weights.exists() else None
    manifest.last_weights = str(last_weights.resolve()) if last_weights.exists() else None
    manifest.valid = bool(manifest.best_weights or manifest.last_weights)

    logger.info("=" * 64)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 64)
    logger.info("Results directory: %s", save_dir)
    logger.info("Best weights:      %s", best_weights)
    return manifest


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main(argv: list[str] | None = None) -> None:
    cli_args, overrides = parse_stage_args(
        "Run the YOLO training stage",
        argv=argv,
        extra_arguments=[
            (("--data",), {"type": str, "default": None}),
            (("--model",), {"type": str, "default": None}),
            (("--epochs",), {"type": int, "default": None}),
            (("--batch",), {"type": int, "default": None}),
            (("--imgsz",), {"type": int, "default": None}),
            (("--patience",), {"type": int, "default": None}),
            (("--device",), {"type": str, "default": None}),
            (("--name",), {"type": str, "default": None}),
            (("--rfs-threshold",), {"dest": "rfs_threshold", "type": float, "default": None}),
            (("--beta",), {"type": float, "default": None}),
            (("--balance",), {"action": "store_true"}),
            (("--no-class-weights",), {"dest": "no_class_weights", "action": "store_true"}),
            (("--dry-run",), {"dest": "dry_run", "action": "store_true"}),
        ],
    )
    cfg = load_cli_config(overrides=overrides)
    if cli_args.run_id:
        cfg.run.run_id = cli_args.run_id
    if cli_args.manifest:
        cfg.training.manifest = cli_args.manifest
    if cli_args.output_root:
        cfg.training.output_root = cli_args.output_root
    if cli_args.data:
        cfg.training.data = cli_args.data
    if cli_args.model:
        cfg.training.model = cli_args.model
    if cli_args.epochs is not None:
        cfg.training.epochs = cli_args.epochs
    if cli_args.batch is not None:
        cfg.training.batch = cli_args.batch
    if cli_args.imgsz is not None:
        cfg.training.imgsz = cli_args.imgsz
    if cli_args.patience is not None:
        cfg.training.patience = cli_args.patience
    if cli_args.device:
        cfg.training.device = cli_args.device
    if cli_args.name:
        cfg.training.name = cli_args.name
    if cli_args.balance:
        cfg.training.balance = True
    if cli_args.rfs_threshold is not None:
        cfg.training.rfs_threshold = cli_args.rfs_threshold
    if cli_args.beta is not None:
        cfg.training.beta = cli_args.beta
    if cli_args.no_class_weights:
        cfg.training.no_class_weights = True
    if cli_args.dry_run:
        cfg.training.dry_run = True

    run_context = create_run_context(cfg, run_id=cfg.run.run_id)
    output_root = (
        ensure_stage_dir(Path(cfg.training.output_root).resolve())
        if cfg.training.output_root
        else ensure_stage_dir(run_context.train_dir)
    )
    training_snapshot = snapshot_config(cfg, output_root / "resolved_config.json")

    data_manifest_path: Optional[Path] = None
    if cfg.training.manifest:
        data_manifest_path = Path(cfg.training.manifest).resolve()
        data_manifest = read_manifest(data_manifest_path, DataManifest)
        verified_data_yaml = Path(data_manifest.yolo_root).resolve() / "data.yaml"
    elif cfg.training.data:
        verified_data_yaml = Path(cfg.training.data).resolve()
    else:
        from yoloml.data.dataset import DatasetManager

        manager = DatasetManager(cfg.dataset)
        prepared = manager.prepare_data(output_root=output_root)
        verified_data_yaml = prepared.data_yaml

    setup_telemetry(cfg.telemetry)
    manifest = train(
        cfg.training,
        verified_data_yaml,
        cfg.telemetry,
        output_root=output_root,
        training_config_snapshot=training_snapshot,
        run_id=run_context.run_id,
        data_manifest_path=data_manifest_path,
    )
    write_manifest(output_root / "train_manifest.json", manifest)

if __name__ == "__main__":
    main()
