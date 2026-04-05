"""
yoloml/models/quantize.py
---------------------------
Multi-level TFLite quantization pipeline for Krishi Vaidya.

Exports a trained YOLOv8 .pt model to TensorFlow Lite at multiple
quantization levels, validates each artifact, and generates a
comparison report suitable for academic publication.

Quantization Levels:
    fp32         Full 32-bit precision (baseline)
    fp16         Half-precision weights (~2× size reduction)
    int8         Full INT8 quantization with calibration data (~4× reduction)

Usage:
    # Single level
    python -m yoloml.models.quantize \\
        --model runs/detect/krishi_bouncer/weights/best.pt \\
        --data krishi_bouncer_dataset/data.yaml \\
        --level int8 --imgsz 416

    # All levels for comparison
    python -m yoloml.models.quantize \\
        --model runs/detect/krishi_bouncer/weights/best.pt \\
        --data krishi_bouncer_dataset/data.yaml \\
        --level all --imgsz 416
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from yoloml.config import setup_config, YoloMLConfig, QuantizationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishi.quantize")

PROJECT_ROOT = Path(__file__).resolve().parents[4]

LEVELS = ("fp32", "fp16", "int8")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_tflite(
    model_path: Path,
    level: str,
    data_yaml: Optional[Path],
    imgsz: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Export a YOLOv8 model to TFLite at the specified quantization level.

    Returns a metadata dict with model_size_bytes, export_time_seconds, etc.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    level_dir = output_dir / level
    level_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model: %s", model_path)
    model = YOLO(str(model_path))

    export_kwargs: Dict[str, Any] = {
        "format": "tflite",
        "imgsz": imgsz,
    }

    if level == "fp16":
        export_kwargs["half"] = True
    elif level == "int8":
        export_kwargs["int8"] = True
        if data_yaml and data_yaml.exists():
            export_kwargs["data"] = str(data_yaml)
            logger.info("Using calibration data: %s", data_yaml)
        else:
            logger.warning(
                "No data.yaml provided for INT8 calibration. "
                "Export may use default calibration or fail."
            )

    logger.info("Exporting to TFLite [%s] at imgsz=%d …", level.upper(), imgsz)
    start = time.time()

    try:
        exported = model.export(**export_kwargs)
    except Exception as exc:
        logger.error("Export failed for level '%s': %s", level, exc)
        return {"level": level, "success": False, "error": str(exc)}

    export_time = time.time() - start
    exported_path = Path(str(exported)) if exported else None

    tflite_path = None
    if exported_path and exported_path.exists() and exported_path.suffix == ".tflite":
        tflite_path = exported_path
    else:
        search_root = model_path.parent
        candidates = list(search_root.rglob("*.tflite"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            tflite_path = candidates[0]

    if tflite_path is None or not tflite_path.exists():
        logger.error("Could not locate exported TFLite file for level '%s'.", level)
        return {"level": level, "success": False, "error": "TFLite file not found after export"}

    final_path = level_dir / f"krishi_bouncer_{level}.tflite"
    shutil.copy2(tflite_path, final_path)
    model_size = final_path.stat().st_size

    logger.info("Exported: %s (%.2f MB, %.1fs)", final_path.name, model_size / 1e6, export_time)

    pt_size = model_path.stat().st_size
    compression = pt_size / max(model_size, 1)

    metadata = {
        "level": level,
        "success": True,
        "tflite_path": str(final_path),
        "model_size_bytes": model_size,
        "model_size_mb": round(model_size / 1e6, 2),
        "pt_size_bytes": pt_size,
        "pt_size_mb": round(pt_size / 1e6, 2),
        "compression_ratio": round(compression, 2),
        "export_time_seconds": round(export_time, 1),
        "imgsz": imgsz,
    }

    meta_path = level_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_tflite(tflite_path: Path) -> Dict[str, Any]:
    """
    Validate that a TFLite file is loadable and inspect its I/O shapes.
    Returns a dict with input/output tensor details.
    """
    try:
        import tensorflow as tf
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            interpreter = tflite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            inputs = interpreter.get_input_details()
            outputs = interpreter.get_output_details()
            return {
                "valid": True,
                "input_shape": [d["shape"].tolist() for d in inputs],
                "output_shape": [d["shape"].tolist() for d in outputs],
                "input_dtype": [str(d["dtype"]) for d in inputs],
                "output_dtype": [str(d["dtype"]) for d in outputs],
            }
        except ImportError:
            logger.warning(
                "Neither tensorflow nor tflite_runtime installed. "
                "Skipping TFLite validation."
            )
            return {"valid": None, "reason": "no_tf_runtime"}

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    return {
        "valid": True,
        "input_shape": [d["shape"].tolist() for d in inputs],
        "output_shape": [d["shape"].tolist() for d in outputs],
        "input_dtype": [str(d["dtype"]) for d in inputs],
        "output_dtype": [str(d["dtype"]) for d in outputs],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_comparison(results: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Generate a comparison report across quantization levels."""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "levels": [],
    }

    baseline_size = None
    for r in results:
        if not r.get("success"):
            report["levels"].append({"level": r["level"], "success": False, "error": r.get("error")})
            continue

        entry = {
            "level": r["level"],
            "success": True,
            "model_size_mb": r["model_size_mb"],
            "compression_ratio": r.get("compression_ratio"),
            "export_time_s": r.get("export_time_seconds"),
            "imgsz": r.get("imgsz"),
        }

        if baseline_size is None:
            baseline_size = r["model_size_bytes"]
        entry["size_vs_fp32"] = round(r["model_size_bytes"] / max(baseline_size, 1), 2)

        tflite_path = Path(r["tflite_path"]) if r.get("tflite_path") else None
        if tflite_path and tflite_path.exists():
            validation = validate_tflite(tflite_path)
            entry["validation"] = validation

        report["levels"].append(entry)

    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Comparison report saved: %s", report_path)

    logger.info("")
    logger.info("╔══════════════╦════════════╦═══════════╦══════════════╗")
    logger.info("║ Level        ║ Size (MB)  ║ vs FP32   ║ Export (s)   ║")
    logger.info("╠══════════════╬════════════╬═══════════╬══════════════╣")
    for entry in report["levels"]:
        if not entry.get("success"):
            logger.info(
                "║ %-12s ║ %-10s ║ %-9s ║ %-12s ║",
                entry["level"], "FAILED", "-", "-",
            )
            continue
        logger.info(
            "║ %-12s ║ %8.2f   ║ %7.2f×  ║ %10.1f   ║",
            entry["level"],
            entry["model_size_mb"],
            entry.get("size_vs_fp32", 0),
            entry.get("export_time_s", 0),
        )
    logger.info("╚══════════════╩════════════╩═══════════╩══════════════╝")

    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

setup_config()

@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: YoloMLConfig) -> None:
    args: QuantizationConfig = cfg.quantization

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    output_dir = (Path(args.output)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data).resolve() if args.data else None

    # Handle "all" logic if somehow levels has "all" in it, otherwise just use the list
    levels = ["fp32", "fp16", "int8"] if "all" in args.levels else args.levels
    if not levels:
        levels = ["fp32", "fp16", "int8"]

    logger.info("=" * 64)
    logger.info("  KRISHI VAIDYA — TFLite Quantization Pipeline")
    logger.info("=" * 64)
    logger.info("Model:       %s", model_path)
    logger.info("Levels:      %s", ", ".join(levels))
    logger.info("Image size:  %d (export)", args.imgsz)
    logger.info("Output:      %s", output_dir)

    results: List[Dict[str, Any]] = []

    for level in levels:
        logger.info("-" * 64)
        logger.info("EXPORTING: %s", level.upper())
        logger.info("-" * 64)
        result = export_tflite(model_path, level, data_yaml, args.imgsz, output_dir)
        results.append(result)

    if len(results) > 1:
        generate_comparison(results, output_dir)

    logger.info("=" * 64)
    logger.info("QUANTIZATION COMPLETE")
    logger.info("=" * 64)
    for r in results:
        if r.get("success"):
            logger.info(
                "  [%s] %s — %.2f MB",
                r["level"].upper(), r.get("tflite_path", ""), r.get("model_size_mb", 0),
            )
        else:
            logger.info("  [%s] FAILED: %s", r["level"].upper(), r.get("error", "unknown"))

if __name__ == "__main__":
    main()
