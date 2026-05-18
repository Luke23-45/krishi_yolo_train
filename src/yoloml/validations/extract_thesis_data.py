"""
yoloml/validations/extract_thesis_data.py
-----------------------------------------
Extract, clean, and restructure all publishable data from a YOLO training run
into a thesis-ready output directory.

Produces:
  <output_dir>/
    metrics/
      thesis_metrics.json          – all epoch metrics (cleaned)
      thesis_metrics.csv           – same, as CSV
      best_epoch_metrics.json      – single best-epoch row
      training_summary.json        – high-level summary (model, epochs, best mAP, etc.)
      per_class_balance.csv        – class distribution + weights
      learning_rate_schedule.csv   – LR per epoch per param-group
      loss_curves.csv              – train + val losses per epoch
      metric_curves.csv            – P, R, mAP50, mAP50-95, F1 per epoch
    metadata/
      balance_report.json          – copied from run
      train_manifest.json          – copied from run
      args.yaml                    – copied from run
      resolved_config.json         – copied from run
    images/
      <qualitative images>         – batch previews, label maps, etc.
"""

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig


def clean_header(header: str) -> str:
    """Clean YOLO results.csv headers by removing whitespaces and standardizing names."""
    return header.strip().replace(" ", "")


def parse_yolo_results(results_csv_path: Path) -> List[Dict[str, Any]]:
    """Parse YOLO results.csv and return a list of dictionaries with clean headers."""
    if not results_csv_path.exists():
        raise FileNotFoundError(f"Could not find {results_csv_path}")

    cleaned_data = []
    with open(results_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = [clean_header(h) for h in next(reader)]

        for row in reader:
            if not row:
                continue
            row_data = {}
            for header, value in zip(headers, row):
                try:
                    row_data[header] = float(value.strip())
                except ValueError:
                    row_data[header] = value.strip()
            cleaned_data.append(row_data)

    return cleaned_data


def extract_best_metrics(data: List[Dict[str, Any]], metric_key: str = "metrics/mAP50-95(B)") -> Dict[str, Any]:
    """Find the best epoch based on a specific metric."""
    if not data:
        return {}

    best_row = max(data, key=lambda x: x.get(metric_key, 0.0))
    return best_row


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    """Write a list of dicts to CSV with automatic fieldname detection."""
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: Any) -> None:
    """Write data to JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def _compute_f1(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if (precision + recall) > 0:
        return 2.0 * precision * recall / (precision + recall)
    return 0.0


def _extract_loss_curves(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract training and validation loss curves as a separate CSV-ready list."""
    loss_keys = [
        "epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
    ]
    rows = []
    for row in data:
        entry: Dict[str, Any] = {}
        for key in loss_keys:
            if key in row:
                entry[key] = row[key]
        # Compute total train and val loss for convenience
        train_total = sum(row.get(k, 0.0) for k in ["train/box_loss", "train/cls_loss", "train/dfl_loss"])
        val_total = sum(row.get(k, 0.0) for k in ["val/box_loss", "val/cls_loss", "val/dfl_loss"])
        entry["train/total_loss"] = round(train_total, 6)
        entry["val/total_loss"] = round(val_total, 6)
        rows.append(entry)
    return rows


def _extract_metric_curves(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract precision, recall, mAP, and F1 curves per epoch."""
    rows = []
    for row in data:
        p = row.get("metrics/precision(B)", 0.0)
        r = row.get("metrics/recall(B)", 0.0)
        entry = {
            "epoch": row.get("epoch", 0),
            "precision": round(float(p), 6),
            "recall": round(float(r), 6),
            "f1": round(_compute_f1(float(p), float(r)), 6),
            "mAP50": round(float(row.get("metrics/mAP50(B)", 0.0)), 6),
            "mAP50_95": round(float(row.get("metrics/mAP50-95(B)", 0.0)), 6),
        }
        rows.append(entry)
    return rows


def _extract_lr_schedule(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract learning rate schedule per epoch across all parameter groups."""
    lr_prefixes = ["lr/"]
    rows = []
    for row in data:
        entry: Dict[str, Any] = {"epoch": row.get("epoch", 0)}
        for key, value in row.items():
            if any(key.startswith(prefix) for prefix in lr_prefixes):
                entry[key] = value
        rows.append(entry)
    return rows


def _extract_class_balance(balance_report: Dict[str, Any], classes_json: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Convert balance report into a per-class CSV-ready list."""
    distribution = balance_report.get("original_distribution", {})
    weights = balance_report.get("class_weights", {})

    # Build class_id -> name mapping from weights keys or classes_json
    class_names: Dict[str, str] = {}
    if classes_json and "names" in classes_json:
        class_names = classes_json["names"]
    else:
        # Infer from weights dict (keys are names, need to map by order)
        weight_names = list(weights.keys())
        for i, name in enumerate(weight_names):
            class_names[str(i)] = name

    rows = []
    for class_id_str, count in sorted(distribution.items(), key=lambda x: int(x[0])):
        class_name = class_names.get(class_id_str, f"class_{class_id_str}")
        weight = weights.get(class_name, None)
        rows.append({
            "class_id": int(class_id_str),
            "class_name": class_name,
            "instance_count": int(count),
            "class_weight": round(float(weight), 6) if weight is not None else None,
        })
    return rows


def _build_training_summary(
    data: List[Dict[str, Any]],
    best_metrics: Dict[str, Any],
    run_dir: Path,
    balance_report: Optional[Dict[str, Any]] = None,
    train_manifest: Optional[Dict[str, Any]] = None,
    resolved_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a high-level training summary JSON for the thesis."""
    total_epochs = len(data)
    last_row = data[-1] if data else {}
    best_epoch = int(best_metrics.get("epoch", 0)) if best_metrics else 0

    # Extract training config details
    training_cfg = {}
    if resolved_config:
        training_cfg = resolved_config.get("training", {})

    summary: Dict[str, Any] = {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "run_id": train_manifest.get("run_id", "unknown") if train_manifest else "unknown",
        "model_architecture": training_cfg.get("model", "unknown"),
        "total_epochs_trained": total_epochs,
        "image_size": training_cfg.get("imgsz", 640),
        "batch_size": training_cfg.get("batch", "unknown"),
        "optimizer": training_cfg.get("optimizer", "unknown"),
        "initial_lr": training_cfg.get("lr0", "unknown"),
        "final_lr_factor": training_cfg.get("lrf", "unknown"),
        "patience": training_cfg.get("patience", "unknown"),
        "best_epoch": best_epoch,
        "best_mAP50": round(float(best_metrics.get("metrics/mAP50(B)", 0.0)), 6) if best_metrics else 0.0,
        "best_mAP50_95": round(float(best_metrics.get("metrics/mAP50-95(B)", 0.0)), 6) if best_metrics else 0.0,
        "best_precision": round(float(best_metrics.get("metrics/precision(B)", 0.0)), 6) if best_metrics else 0.0,
        "best_recall": round(float(best_metrics.get("metrics/recall(B)", 0.0)), 6) if best_metrics else 0.0,
        "best_f1": round(_compute_f1(
            float(best_metrics.get("metrics/precision(B)", 0.0)),
            float(best_metrics.get("metrics/recall(B)", 0.0)),
        ), 6) if best_metrics else 0.0,
        "final_train_box_loss": round(float(last_row.get("train/box_loss", 0.0)), 6),
        "final_train_cls_loss": round(float(last_row.get("train/cls_loss", 0.0)), 6),
        "final_train_dfl_loss": round(float(last_row.get("train/dfl_loss", 0.0)), 6),
        "final_val_box_loss": round(float(last_row.get("val/box_loss", 0.0)), 6),
        "final_val_cls_loss": round(float(last_row.get("val/cls_loss", 0.0)), 6),
        "final_val_dfl_loss": round(float(last_row.get("val/dfl_loss", 0.0)), 6),
        "total_training_time_seconds": round(float(last_row.get("time", 0.0)), 1),
        "number_of_classes": balance_report.get("class_weights", {}) and len(balance_report.get("class_weights", {})) or 0 if balance_report else 0,
        "class_balancing_enabled": balance_report.get("rfs_enabled", False) if balance_report else False,
        "beta": balance_report.get("beta", None) if balance_report else None,
    }
    return summary


@hydra.main(version_base=None, config_path="../../../configs/validation", config_name="extract_thesis_data")
def main(cfg: DictConfig):
    run_dir = Path(cfg.run_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve()

    if not run_dir.exists():
        print(f"Error: Run directory {run_dir} does not exist.")
        return

    # Create structured output subdirectories
    metrics_dir = output_dir / "metrics"
    metadata_dir = output_dir / "metadata"
    images_dir = output_dir / "images"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Parse results.csv (check run_dir and run_dir/artifacts)
    # --------------------------------------------------------------------------
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        results_csv = run_dir / "artifacts" / "results.csv"

    clean_data: List[Dict[str, Any]] = []
    best_metrics: Dict[str, Any] = {}

    if results_csv.exists():
        print(f"Parsing {results_csv}...")
        clean_data = parse_yolo_results(results_csv)

        # Save as clean JSON (all epochs)
        _write_json(metrics_dir / "thesis_metrics.json", clean_data)

        # Save as clean CSV (all epochs)
        _write_csv(metrics_dir / "thesis_metrics.csv", clean_data)

        # Extract best metrics
        best_metrics = extract_best_metrics(clean_data)
        if best_metrics:
            _write_json(metrics_dir / "best_epoch_metrics.json", best_metrics)

        # Extract loss curves
        loss_curves = _extract_loss_curves(clean_data)
        if loss_curves:
            _write_csv(metrics_dir / "loss_curves.csv", loss_curves)

        # Extract metric curves (P, R, F1, mAP)
        metric_curves = _extract_metric_curves(clean_data)
        if metric_curves:
            _write_csv(metrics_dir / "metric_curves.csv", metric_curves)

        # Extract learning rate schedule
        lr_schedule = _extract_lr_schedule(clean_data)
        if lr_schedule:
            _write_csv(metrics_dir / "learning_rate_schedule.csv", lr_schedule)

        print(f"  -> Exported {len(clean_data)} epoch rows to metrics/")
    else:
        print(f"Warning: results.csv not found in {run_dir} or {run_dir / 'artifacts'}. Skipping metrics extraction.")

    # --------------------------------------------------------------------------
    # 2. Copy metadata files (balance_report, train_manifest, args.yaml, resolved_config)
    # --------------------------------------------------------------------------
    meta_files = ["balance_report.json", "train_manifest.json", "args.yaml", "resolved_config.json"]
    balance_report: Optional[Dict[str, Any]] = None
    train_manifest: Optional[Dict[str, Any]] = None
    resolved_config: Optional[Dict[str, Any]] = None

    for meta_file in meta_files:
        src_path = run_dir / meta_file
        if not src_path.exists():
            src_path = run_dir / "artifacts" / meta_file
        if src_path.exists():
            shutil.copy2(src_path, metadata_dir / meta_file)
            print(f"  -> Copied metadata: {meta_file}")

            # Load for summary generation
            if meta_file == "balance_report.json":
                balance_report = json.loads(src_path.read_text(encoding="utf-8"))
            elif meta_file == "train_manifest.json":
                train_manifest = json.loads(src_path.read_text(encoding="utf-8"))
            elif meta_file == "resolved_config.json":
                resolved_config = json.loads(src_path.read_text(encoding="utf-8"))
        else:
            print(f"  -> Warning: {meta_file} not found. Skipping.")

    # --------------------------------------------------------------------------
    # 3. Generate per-class balance CSV
    # --------------------------------------------------------------------------
    if balance_report:
        classes_json = None
        # Try to find dataset root from resolved config first
        dataset_root = None
        if resolved_config:
            yolo_root = resolved_config.get("dataset", {}).get("yolo_root")
            if yolo_root:
                dataset_root = Path(yolo_root).resolve()
        
        # Fallback: traverse up looking for classes.json
        if not dataset_root or not dataset_root.exists():
            current = run_dir
            while current.parent != current:
                potential = current / "krishi_bouncer_dataset" / "classes.json"
                if potential.exists():
                    classes_json = json.loads(potential.read_text(encoding="utf-8"))
                    break
                current = current.parent
        else:
            classes_path = dataset_root / "classes.json"
            if classes_path.exists():
                classes_json = json.loads(classes_path.read_text(encoding="utf-8"))

        class_balance_rows = _extract_class_balance(balance_report, classes_json)
        if class_balance_rows:
            _write_csv(metrics_dir / "per_class_balance.csv", class_balance_rows)
            print(f"  -> Exported per-class balance ({len(class_balance_rows)} classes)")

    # --------------------------------------------------------------------------
    # 4. Generate training summary
    # --------------------------------------------------------------------------
    if clean_data:
        summary = _build_training_summary(
            clean_data, best_metrics, run_dir,
            balance_report, train_manifest, resolved_config,
        )
        _write_json(metrics_dir / "training_summary.json", summary)
        print("  -> Generated training_summary.json")

    # --------------------------------------------------------------------------
    # 5. Extract qualitative images (exclude auto-generated plots that will be
    #    regenerated with thesis-quality styling)
    # --------------------------------------------------------------------------
    print("Extracting qualitative image data...")
    excluded_images = {
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxF1_curve.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
    }

    image_count = 0
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for search_dir in [run_dir, run_dir / "artifacts"]:
        if not search_dir.exists():
            continue
        for img_path in search_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in image_suffixes and img_path.name not in excluded_images:
                shutil.copy2(img_path, images_dir / img_path.name)
                image_count += 1

    print(f"  -> Copied {image_count} qualitative images to images/")

    # --------------------------------------------------------------------------
    # 6. Write extraction manifest (so we know what was produced)
    # --------------------------------------------------------------------------
    extraction_manifest = {
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "files_produced": {
            "metrics": sorted([f.name for f in metrics_dir.iterdir() if f.is_file()]),
            "metadata": sorted([f.name for f in metadata_dir.iterdir() if f.is_file()]),
            "images": sorted([f.name for f in images_dir.iterdir() if f.is_file()]),
        },
        "total_epochs": len(clean_data),
        "best_epoch": int(best_metrics.get("epoch", 0)) if best_metrics else None,
        "qualitative_images_count": image_count,
    }
    _write_json(output_dir / "extraction_manifest.json", extraction_manifest)

    print(f"\nExtraction complete. Thesis-ready data is in: {output_dir}")
    print(f"  metrics/  : {len(list(metrics_dir.iterdir()))} files")
    print(f"  metadata/ : {len(list(metadata_dir.iterdir()))} files")
    print(f"  images/   : {image_count} files")


if __name__ == "__main__":
    main()
