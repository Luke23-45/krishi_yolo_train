"""
yoloml/validations/benchmark_latency.py
---------------------------------------
Benchmark inference latency of a trained YOLO model on a set of images.

Produces (inside <output_dir>/):
    latency_metrics.json           – aggregate statistics (mean, median, p95, p99, fps, etc.)
    latency_per_image.csv          – per-image latency breakdown
    latency_summary.csv            – single-row CSV of aggregate stats (for easy table inclusion)
"""

from __future__ import annotations

import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig


def collect_images(images_dir: Path) -> List[Path]:
    """Collect image paths for benchmarking."""
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in valid_suffixes])


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


@hydra.main(version_base=None, config_path="../../../configs/validation", config_name="benchmark_latency")
def main(cfg: DictConfig):

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Benchmarking requires ultralytics to be installed.") from exc

    # Resolve paths
    images_dir = Path(cfg.images_dir).resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = collect_images(images_dir)
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    output_dir = Path(cfg.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = str(Path(cfg.model).resolve())
    print(f"Loading model from {model_path} on {cfg.device}...")
    model = YOLO(model_path)

    # Warmup
    warmup_count = int(cfg.get("warmup", 10))
    print(f"Warming up for {warmup_count} iterations...")
    warmup_img = str(image_paths[0])
    for _ in range(warmup_count):
        _ = model.predict(source=warmup_img, imgsz=cfg.imgsz, device=cfg.device, verbose=False)

    # Benchmark
    print(f"Benchmarking inference on {len(image_paths)} images...")
    per_image_rows: List[Dict[str, Any]] = []

    for img_path in image_paths:
        start_time = time.perf_counter()
        results = model.predict(source=str(img_path), imgsz=cfg.imgsz, device=cfg.device, verbose=False)
        end_time = time.perf_counter()

        latency = (end_time - start_time) * 1000.0
        num_detections = 0
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            num_detections = len(results[0].boxes)

        per_image_rows.append({
            "image_name": img_path.name,
            "image_path": str(img_path),
            "latency_ms": round(latency, 3),
            "num_detections": num_detections,
        })

    latencies_ms = [row["latency_ms"] for row in per_image_rows]

    # Compute statistics
    avg_latency = float(np.mean(latencies_ms))
    median_latency = float(np.median(latencies_ms))
    std_latency = float(np.std(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))
    p99_latency = float(np.percentile(latencies_ms, 99))
    min_latency = float(np.min(latencies_ms))
    max_latency = float(np.max(latencies_ms))
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    # Build aggregate metrics
    metrics: Dict[str, Any] = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "device": cfg.device,
        "image_size": cfg.imgsz,
        "warmup_iterations": warmup_count,
        "total_images_tested": len(image_paths),
        "images_dir": str(images_dir),
        "system_info": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        },
        "fps": round(fps, 2),
        "latency_ms": {
            "mean": round(avg_latency, 3),
            "median": round(median_latency, 3),
            "std": round(std_latency, 3),
            "min": round(min_latency, 3),
            "max": round(max_latency, 3),
            "p95": round(p95_latency, 3),
            "p99": round(p99_latency, 3),
        },
    }

    # Write JSON
    _write_json(output_dir / "latency_metrics.json", metrics)

    # Write per-image CSV
    _write_csv(
        output_dir / "latency_per_image.csv",
        per_image_rows,
        ["image_name", "image_path", "latency_ms", "num_detections"],
    )

    # Write single-row summary CSV (easy to paste into thesis tables)
    summary_row = {
        "model": Path(model_path).name,
        "device": cfg.device,
        "imgsz": cfg.imgsz,
        "num_images": len(image_paths),
        "fps": round(fps, 2),
        "mean_ms": round(avg_latency, 3),
        "median_ms": round(median_latency, 3),
        "std_ms": round(std_latency, 3),
        "min_ms": round(min_latency, 3),
        "max_ms": round(max_latency, 3),
        "p95_ms": round(p95_latency, 3),
        "p99_ms": round(p99_latency, 3),
    }
    _write_csv(
        output_dir / "latency_summary.csv",
        [summary_row],
        list(summary_row.keys()),
    )

    print(f"\nBenchmarking complete.")
    print(f"  Average FPS      : {fps:.2f}")
    print(f"  Mean latency     : {avg_latency:.3f} ms")
    print(f"  Median latency   : {median_latency:.3f} ms")
    print(f"  Std deviation    : {std_latency:.3f} ms")
    print(f"  P95 latency      : {p95_latency:.3f} ms")
    print(f"  Results saved to : {output_dir}")


if __name__ == "__main__":
    main()
