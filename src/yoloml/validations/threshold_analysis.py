"""
yoloml/validations/threshold_analysis.py
----------------------------------------
Sweep confidence thresholds and record detection metrics at each level.

Produces (inside <output_dir>/):
    threshold_sweep.csv            – per-threshold P, R, F1, mAP50, mAP50-95, fitness
    threshold_sweep.json           – same data as JSON (for programmatic analysis)
    optimal_thresholds.json        – best threshold for each metric (F1, mAP50, mAP50-95)
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import hydra
from omegaconf import DictConfig


def _write_json(path: Path, data: Any) -> None:
    """Write data to JSON with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write a list of dicts to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_optimal(results: List[Dict[str, Any]], metric_key: str) -> Dict[str, Any]:
    """Find the threshold that maximises a given metric."""
    if not results:
        return {}
    best = max(results, key=lambda r: r.get(metric_key, 0.0))
    return {
        "metric": metric_key,
        "optimal_threshold": best["confidence_threshold"],
        "value": best[metric_key],
    }


@hydra.main(version_base=None, config_path="../../../configs/validation", config_name="threshold_analysis")
def main(cfg: DictConfig):

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Threshold analysis requires ultralytics to be installed.") from exc

    model_path = str(Path(cfg.model).resolve())
    print(f"Loading model from {model_path} on {cfg.device}...")
    model = YOLO(model_path)

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data: List[Dict[str, Any]] = []

    # Generate thresholds avoiding floating point errors
    thresholds = []
    current = float(cfg.start)
    end = float(cfg.end)
    step = float(cfg.step)
    while current <= end + 1e-9:
        thresholds.append(round(current, 3))
        current += step

    print(f"Sweeping {len(thresholds)} thresholds: {thresholds}")

    for conf in thresholds:
        print(f"\nEvaluating with confidence threshold = {conf}...")

        try:
            metrics = model.val(
                data=cfg.data,
                imgsz=cfg.imgsz,
                device=cfg.device,
                conf=conf,
                verbose=False,
                plots=False,
                save_json=False,
            )

            # Extract metrics from results_dict
            precision = float(metrics.results_dict.get("metrics/precision(B)", 0.0))
            recall = float(metrics.results_dict.get("metrics/recall(B)", 0.0))
            map50 = float(metrics.results_dict.get("metrics/mAP50(B)", 0.0))
            map50_95 = float(metrics.results_dict.get("metrics/mAP50-95(B)", 0.0))
            fitness = float(metrics.fitness) if hasattr(metrics, "fitness") else 0.0

            # Compute F1 Score
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            results_data.append({
                "confidence_threshold": conf,
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1_score": round(f1, 6),
                "mAP_50": round(map50, 6),
                "mAP_50_95": round(map50_95, 6),
                "fitness": round(fitness, 6),
            })

            print(f"  Conf={conf} | P={precision:.4f} | R={recall:.4f} | F1={f1:.4f} | mAP50-95={map50_95:.4f}")

        except Exception as e:
            print(f"  Warning: Failed to extract metrics for threshold {conf}: {e}")
            results_data.append({
                "confidence_threshold": conf,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "mAP_50": None,
                "mAP_50_95": None,
                "fitness": None,
            })

    if not any(r.get("precision") is not None for r in results_data):
        print("Error: No valid data was collected across any threshold.")
        return

    # Write CSV
    fieldnames = ["confidence_threshold", "precision", "recall", "f1_score", "mAP_50", "mAP_50_95", "fitness"]
    _write_csv(output_dir / "threshold_sweep.csv", results_data, fieldnames)

    # Write JSON
    sweep_output = {
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "data_yaml": str(Path(cfg.data).resolve()),
        "device": cfg.device,
        "image_size": cfg.imgsz,
        "thresholds_tested": thresholds,
        "results": results_data,
    }
    _write_json(output_dir / "threshold_sweep.json", sweep_output)

    # Find optimal thresholds
    valid_results = [r for r in results_data if r.get("precision") is not None]
    optimal = {
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "optimal_f1": _find_optimal(valid_results, "f1_score"),
        "optimal_mAP50": _find_optimal(valid_results, "mAP_50"),
        "optimal_mAP50_95": _find_optimal(valid_results, "mAP_50_95"),
        "optimal_precision": _find_optimal(valid_results, "precision"),
        "optimal_recall": _find_optimal(valid_results, "recall"),
    }
    _write_json(output_dir / "optimal_thresholds.json", optimal)

    print(f"\nThreshold analysis complete.")
    print(f"  Results saved to: {output_dir}")
    print(f"  Best F1 threshold: {optimal['optimal_f1'].get('optimal_threshold', 'N/A')} "
          f"(F1={optimal['optimal_f1'].get('value', 0.0):.4f})")
    print(f"  Best mAP50-95 threshold: {optimal['optimal_mAP50_95'].get('optimal_threshold', 'N/A')} "
          f"(mAP50-95={optimal['optimal_mAP50_95'].get('value', 0.0):.4f})")


if __name__ == "__main__":
    main()
