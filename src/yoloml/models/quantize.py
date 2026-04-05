"""
yoloml/models/quantize.py
---------------------------
Multi-level TFLite quantization pipeline for Krishi Vaidya.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from yoloml.config import QuantizationConfig
from yoloml.pipeline import (
    QuantManifest,
    TrainManifest,
    create_run_context,
    ensure_stage_dir,
    load_cli_config,
    parse_stage_args,
    read_manifest,
    snapshot_config,
    write_manifest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishi.quantize")


def export_tflite(
    model_path: Path,
    level: str,
    data_yaml: Optional[Path],
    imgsz: int,
    output_dir: Path,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    level_dir = output_dir / level
    level_dir.mkdir(parents=True, exist_ok=True)

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
        else:
            logger.warning("No data.yaml provided for INT8 calibration.")

    start = time.time()
    try:
        exported = model.export(**export_kwargs)
    except Exception as exc:
        logger.error("Export failed for level '%s': %s", level, exc)
        return {"level": level, "success": False, "error": str(exc)}

    export_time = time.time() - start
    exported_path = Path(str(exported)) if exported else None
    tflite_path = exported_path if exported_path and exported_path.exists() and exported_path.suffix == ".tflite" else None
    if tflite_path is None:
        candidates = sorted(model_path.parent.rglob("*.tflite"), key=lambda path: path.stat().st_mtime, reverse=True)
        if candidates:
            tflite_path = candidates[0]
    if tflite_path is None or not tflite_path.exists():
        return {"level": level, "success": False, "error": "TFLite file not found after export"}

    final_path = level_dir / f"krishi_bouncer_{level}.tflite"
    shutil.copy2(tflite_path, final_path)
    model_size = final_path.stat().st_size
    pt_size = model_path.stat().st_size
    metadata = {
        "level": level,
        "success": True,
        "tflite_path": str(final_path.resolve()),
        "model_size_bytes": model_size,
        "model_size_mb": round(model_size / 1e6, 2),
        "pt_size_bytes": pt_size,
        "pt_size_mb": round(pt_size / 1e6, 2),
        "compression_ratio": round(pt_size / max(model_size, 1), 2),
        "export_time_seconds": round(export_time, 1),
        "imgsz": imgsz,
    }
    meta_path = level_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(meta_path.resolve())
    return metadata


def validate_tflite(tflite_path: Path) -> Dict[str, Any]:
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


def generate_comparison(results: List[Dict[str, Any]], output_dir: Path) -> Optional[Path]:
    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "levels": []}
    baseline_size = None
    for result in results:
        if not result.get("success"):
            report["levels"].append({"level": result["level"], "success": False, "error": result.get("error")})
            continue
        if baseline_size is None:
            baseline_size = result["model_size_bytes"]
        entry = {
            "level": result["level"],
            "success": True,
            "model_size_mb": result["model_size_mb"],
            "compression_ratio": result["compression_ratio"],
            "export_time_s": result["export_time_seconds"],
            "size_vs_fp32": round(result["model_size_bytes"] / max(baseline_size, 1), 2),
            "validation": validate_tflite(Path(result["tflite_path"])),
        }
        report["levels"].append(entry)
    report_path = output_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _resolve_quant_inputs(args: QuantizationConfig) -> tuple[Path, Optional[Path], Optional[Path]]:
    train_manifest_path: Optional[Path] = None
    if args.manifest:
        train_manifest_path = Path(args.manifest).resolve()
        manifest = read_manifest(train_manifest_path, TrainManifest)
        if not manifest.valid:
            raise RuntimeError(f"Train manifest is not valid: {train_manifest_path}")
        model_path = Path(manifest.best_weights or manifest.last_weights or "")
        data_yaml = Path(manifest.verified_data_yaml) if manifest.verified_data_yaml else None
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found from train manifest: {model_path}")
        return model_path.resolve(), data_yaml.resolve() if data_yaml else None, train_manifest_path

    if not args.model:
        raise ValueError("Quantization requires either --manifest or quantization.model")
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    data_yaml = Path(args.data).resolve() if args.data else None
    return model_path, data_yaml, None


def main(argv: list[str] | None = None) -> None:
    cli_args, overrides = parse_stage_args("Quantize trained YOLO weights to TFLite", argv=argv)
    cfg = load_cli_config(overrides=overrides)
    if cli_args.run_id:
        cfg.run.run_id = cli_args.run_id
    if cli_args.manifest:
        cfg.quantization.manifest = cli_args.manifest
    if cli_args.output_root:
        cfg.quantization.output_root = cli_args.output_root

    run_context = create_run_context(cfg, run_id=cfg.run.run_id)
    output_root = (
        ensure_stage_dir(Path(cfg.quantization.output_root).resolve())
        if cfg.quantization.output_root
        else ensure_stage_dir(run_context.quantize_dir)
    )
    snapshot_config(cfg, output_root / "resolved_config.json")

    args: QuantizationConfig = cfg.quantization
    model_path, data_yaml, train_manifest_path = _resolve_quant_inputs(args)
    levels = ["fp32", "fp16", "int8"] if "all" in args.levels else list(args.levels)
    if not levels:
        levels = ["fp32", "fp16", "int8"]

    results = [export_tflite(model_path, level, data_yaml, args.imgsz, output_root) for level in levels]
    comparison_report = generate_comparison(results, output_root) if len(results) > 1 else None
    manifest = QuantManifest(
        run_id=run_context.run_id,
        train_manifest=str(train_manifest_path) if train_manifest_path else None,
        source_model=str(model_path.resolve()),
        data_yaml=str(data_yaml.resolve()) if data_yaml else None,
        output_root=str(output_root.resolve()),
        levels=results,
        comparison_report=str(comparison_report.resolve()) if comparison_report else None,
        valid=all(result.get("success", False) for result in results),
    )
    write_manifest(output_root / "quant_manifest.json", manifest)


if __name__ == "__main__":
    main()
