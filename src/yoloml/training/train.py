"""
yoloml/training/train.py
-------------------------
End-to-end YOLOv8 training pipeline with class-imbalance mitigation.

Implements two complementary strategies:

1. Offline Repeat Factor Sampling (RFS)
   Gupta et al., "LVIS: A Dataset for Large Vocabulary Instance Segmentation", CVPR 2019
   Creates a rebalanced copy of the training set by repeating images that contain rare classes.

2. Class-Balanced Loss Weighting
   Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019
   Computes per-class weights via the effective-number formula and injects them
   into the model's classification loss before training begins.

Usage:
    python -m yoloml.training.train
    python -m yoloml.training.train training.epochs=200 training.batch=32
    python -m yoloml.training.train training.dry_run=true
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from yoloml.config import TelemetryConfig, TrainingConfig, load_config
from yoloml.pipeline import (
    DataManifest,
    TrainManifest,
    create_run_context,
    ensure_stage_dir,
    read_manifest,
    snapshot_config,
    write_manifest,
)
from yoloml.utils.provisioning import ensure_dataset_ready
from yoloml.utils.telemetry import setup_telemetry

logger = logging.getLogger("yoloml.train")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def get_labels_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    for i in reversed(range(len(parts))):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts)


def scan_class_distribution(label_dir: Path) -> tuple[Counter, dict[str, set[int]]]:
    class_counts: Counter = Counter()
    image_classes: dict[str, set[int]] = {}

    for label_path in sorted(label_dir.iterdir()):
        if not label_path.is_file() or label_path.suffix.lower() != ".txt":
            continue
        stem = label_path.stem
        classes_in_image: set[int] = set()
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


def effective_number(n: float, beta: float) -> float:
    if abs(beta - 1.0) < 1e-12:
        return float(n)
    if beta < 1e-12:
        return 1.0
    return (1.0 - beta ** n) / (1.0 - beta)


def compute_repeat_factors(
    image_classes: dict[str, set[int]],
    threshold: float | None = None,
) -> dict[str, int]:
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

    repeat_factors: dict[str, int] = {}
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
    repeat_factors: dict[str, int],
    yolo_cfg: dict[str, Any],
) -> dict[str, int]:
    train_rel = yolo_cfg.get("train", "images/train")
    if isinstance(train_rel, list):
        train_rel = train_rel[0]

    src_images = (source_dir / train_rel).resolve()
    src_labels = get_labels_dir(src_images)
    dst_images = (output_dir / train_rel).resolve()
    dst_labels = get_labels_dir(dst_images)

    for d in (dst_images, dst_labels):
        d.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        split_rel = yolo_cfg.get(split)
        if not split_rel:
            continue
        if isinstance(split_rel, list):
            split_rel = split_rel[0]

        src_split_img = (source_dir / split_rel).resolve()
        if not src_split_img.exists():
            continue

        dst_split_img = (output_dir / split_rel).resolve()
        dst_split_lbl = get_labels_dir(dst_split_img)

        if dst_split_img.exists():
            shutil.rmtree(dst_split_img)
        shutil.copytree(src_split_img, dst_split_img)

        src_split_lbl = get_labels_dir(src_split_img)
        if src_split_lbl.exists():
            if dst_split_lbl.exists():
                shutil.rmtree(dst_split_lbl)
            shutil.copytree(src_split_lbl, dst_split_lbl)

    def _link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            os.link(src, dst)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)

    available_images: dict[str, Path] = {}
    if src_images.exists():
        for img_file in src_images.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                available_images[img_file.stem] = img_file

    duplicated = 0
    total = 0

    for stem, factor in repeat_factors.items():
        src_label = src_labels / f"{stem}.txt"
        if not src_label.exists():
            continue

        src_image = available_images.get(stem)
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

    cfg = yolo_cfg.copy()
    cfg["path"] = str(output_dir.resolve())
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

    return {
        "original_images": len(repeat_factors),
        "balanced_images": total,
        "duplicated_images": duplicated,
    }


def compute_class_weights(
    class_counts: dict[int, int],
    num_classes: int,
    beta: float = 0.9999,
) -> list[float]:
    eff_nums = []
    for c in range(num_classes):
        n = class_counts.get(c, 1)
        eff_nums.append(effective_number(max(n, 1), beta))

    inv_eff = [1.0 / e for e in eff_nums]
    total = sum(inv_eff)
    weights = [(w / total) * num_classes for w in inv_eff]
    return weights


def inject_class_weights(model: Any, weights: list[float]) -> bool:
    try:
        import torch
        from ultralytics.utils.loss import v8DetectionLoss

        weight_tensor = torch.tensor(weights, dtype=torch.float32)

        original_init = v8DetectionLoss.__init__

        def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            self.bce = torch.nn.BCEWithLogitsLoss(
                pos_weight=weight_tensor.to(self.device),
                reduction="none",
            )

        v8DetectionLoss.__init__ = patched_init

        logger.info("Injected class-balanced weights via monkey-patching v8DetectionLoss.")
        return True

    except Exception as exc:
        logger.warning("Could not inject class-balanced weights: %s", exc)
        return False


def resolve_device(requested: str) -> str:
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
    data_manifest_path: Path | None = None,
) -> TrainManifest:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    dataset_dir = data_yaml.parent

    with open(data_yaml, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    num_classes = int(cfg["nc"])
    if isinstance(cfg["names"], list):
        names = dict(enumerate(cfg["names"]))
    else:
        names = {int(k): v for k, v in cfg["names"].items()}

    logger.info("=" * 64)
    logger.info("  YOLOML - Training Pipeline v1.0")
    logger.info("=" * 64)
    logger.info("Dataset:     %s", dataset_dir)
    logger.info("Classes:     %d", num_classes)
    logger.info("Model:       %s", args.model)
    logger.info("Epochs:      %d", args.epochs)
    logger.info("Batch:       %d", args.batch)
    logger.info("Image size:  %d", args.imgsz)
    logger.info("Device:      %s", args.device)
    logger.info("Patience:    %d", args.patience)
    logger.info("-" * 64)
    logger.info("CLASS IMBALANCE MITIGATION:")
    logger.info("  Balance:    %s", "enabled" if args.balance else "disabled")
    logger.info("  RFS threshold: %s", args.rfs_threshold or "auto")
    logger.info("  Beta:       %.4f", args.beta)
    logger.info("-" * 64)

    train_rel = cfg.get("train", "images/train")
    if isinstance(train_rel, list):
        train_rel = train_rel[0]

    label_dir = get_labels_dir((dataset_dir / train_rel).resolve())
    if not label_dir.exists():
        raise FileNotFoundError(f"Training labels not found: {label_dir}")

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

        is_balanced_valid = False
        if balanced_dir.exists():
            balanced_yaml = balanced_dir / "data.yaml"
            if balanced_yaml.exists():
                balanced_train_img = (balanced_dir / train_rel).resolve()
                if balanced_train_img.exists() and any(balanced_train_img.iterdir()):
                    is_balanced_valid = True
                    logger.info("Existing balanced dataset detected at %s. Skipping regeneration.", balanced_dir)

        if not is_balanced_valid:
            if balanced_dir.exists():
                shutil.rmtree(balanced_dir)
            logger.info("Creating balanced dataset at: %s", balanced_dir)
            stats = create_balanced_dataset(dataset_dir, balanced_dir, repeat_factors, cfg)
            logger.info(
                "  Original: %d images -> Balanced: %d images (+%d duplicates)",
                stats["original_images"], stats["balanced_images"], stats["duplicated_images"],
            )

        balanced_label_dir = get_labels_dir((balanced_dir / train_rel).resolve())
        balanced_counts, _ = scan_class_distribution(balanced_label_dir)
        logger.info("  Post-RFS distribution:")
        for cid in sorted(names.keys()):
            orig = class_counts.get(cid, 0)
            balanced = balanced_counts.get(cid, 0)
            ratio = balanced / max(orig, 1)
            logger.info(
                "    %2d | %-20s | %7d -> %7d (x%.1f)",
                cid, names[cid], orig, balanced, ratio,
            )

        training_data_yaml = balanced_dir / "data.yaml"

    logger.info("-" * 64)
    logger.info("STEP 3: Computing class-balanced loss weights (Cui et al., CVPR 2019)")
    logger.info("-" * 64)
    weights = compute_class_weights(class_counts, num_classes, beta=args.beta)
    for cid in sorted(names.keys()):
        logger.info("  %2d | %-20s | weight = %.4f", cid, names[cid], weights[cid])

    balance_report: dict[str, Any] = {
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
    except ImportError as exc:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics") from exc

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

    kwargs: dict[str, Any] = {
        "data": str(training_data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "patience": args.patience,
        "exist_ok": True,
        "model": args.model,
        "project": str(output_root),
        "name": "artifacts",
        
        # Advanced hyperparameters
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "warmup_momentum": args.warmup_momentum,
        "warmup_bias_lr": args.warmup_bias_lr,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "close_mosaic": args.close_mosaic,
        "auto": args.auto,
        "fraction": args.fraction,
        "val": args.val,
        "save_period": args.save_period,
        "cache": args.cache,
        "rect": args.rect,
        "single_cls": args.single_cls,
        "plots": args.plots,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "workers": args.workers,
    }

    if args.name:
        kwargs["name"] = args.name

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


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(overrides=argv)

    run_context = create_run_context(cfg, run_id=cfg.run.run_id)
    output_root = (
        Path(cfg.training.output_root).resolve()
        if cfg.training.output_root
        else run_context.train_dir
    )
    ensure_stage_dir(output_root)
    training_snapshot = snapshot_config(cfg, output_root / "resolved_config.json")

    data_manifest_path: Path | None = None
    if cfg.training.manifest:
        data_manifest_path = Path(cfg.training.manifest).resolve()
        data_manifest = read_manifest(data_manifest_path, DataManifest)
        verified_data_yaml = Path(data_manifest.yolo_root).resolve() / "data.yaml"
    elif cfg.training.data:
        verified_data_yaml = Path(cfg.training.data).resolve()
    else:
        verified_data_yaml = ensure_dataset_ready(cfg)

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
    main(sys.argv[1:] if len(sys.argv) > 1 else None)
