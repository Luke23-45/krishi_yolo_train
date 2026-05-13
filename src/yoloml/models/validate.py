"""
yoloml/models/validate.py
-------------------------
Validate exported model artifacts from the quantization pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from yoloml.config import ModelValidationConfig, load_config
from yoloml.models.quantize import validate_tflite
from yoloml.pipeline import (
    ModelValidationManifest,
    QuantManifest,
    TrainManifest,
    create_run_context,
    ensure_stage_dir,
    read_manifest,
    snapshot_config,
    write_manifest,
)

logger = logging.getLogger("yoloml.validate")


def run_benchmark(tflite_path: Path, data_yaml: str, args: ModelValidationConfig) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        logger.warning("ultralytics not installed, skipping benchmark.")
        return {"error": "ultralytics_not_installed"}

    logger.info("Benchmarking %s on %s...", tflite_path.name, data_yaml)
    try:
        model = YOLO(str(tflite_path), task="detect")
        results = model.val(
            data=data_yaml,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            plots=False,
            verbose=False,
        )
        return {
            "mAP50-95": round(float(results.box.map), 4),
            "mAP50": round(float(results.box.map50), 4),
            "precision": round(float(results.box.mp), 4),
            "recall": round(float(results.box.mr), 4),
            "inference_time_ms_per_img": round(float(sum(results.speed.values())), 2) if hasattr(results, "speed") else None,
        }
    except Exception as exc:
        logger.error("Benchmark failed for %s: %s", tflite_path.name, exc)
        return {"error": str(exc)}



def main(argv: list[str] | None = None) -> None:
    cfg = load_config(overrides=argv)
    run_context = create_run_context(cfg, run_id=cfg.run.run_id)
    output_root = (
        Path(cfg.model_validation.output_root).resolve()
        if cfg.model_validation.output_root
        else run_context.model_validation_dir
    )
    ensure_stage_dir(output_root)
    snapshot_config(cfg, output_root / "resolved_config.json")

    args: ModelValidationConfig = cfg.model_validation
    if not args.manifest:
        raise ValueError("Model validation requires model_validation.manifest pointing to quant_manifest.json")

    quant_manifest_path = Path(args.manifest).resolve()
    quant_manifest = read_manifest(quant_manifest_path, QuantManifest)
    if not quant_manifest.valid:
        raise RuntimeError(f"Quantization manifest is not valid: {quant_manifest_path}")

    data_yaml = args.data
    if not data_yaml and quant_manifest.data_yaml:
        data_yaml = quant_manifest.data_yaml
    elif not data_yaml and quant_manifest.train_manifest:
        try:
            train_manifest = read_manifest(quant_manifest.train_manifest, TrainManifest)
            data_yaml = train_manifest.verified_data_yaml
        except Exception:
            pass

    if args.benchmark and not data_yaml:
        logger.warning("Benchmark enabled but no data.yaml found. Skipping benchmark.")
        args.benchmark = False

    validated_artifacts: list[dict[str, object]] = []
    valid = True
    for level in quant_manifest.levels:
        artifact: dict[str, object] = {
            "level": str(level.get("level", "")),
            "success": bool(level.get("success", False)),
            "tflite_path": level.get("tflite_path"),
        }
        tflite_path = Path(level["tflite_path"]) if level.get("tflite_path") else None
        if level.get("success") and tflite_path and tflite_path.exists():
            artifact["validation"] = validate_tflite(tflite_path)
            validation_status = artifact["validation"].get("valid", False)
            if validation_status is None:
                artifact["valid"] = not args.require_runtime
                if args.require_runtime:
                    artifact["validation"]["reason"] = "runtime_required_but_unavailable"
            else:
                artifact["valid"] = bool(validation_status)
            
            if artifact["valid"] and args.benchmark and data_yaml:
                artifact["benchmark"] = run_benchmark(tflite_path, data_yaml, args)
        else:
            artifact["validation"] = {"valid": False, "reason": level.get("error", "missing_artifact")}
            artifact["valid"] = False
        valid = valid and bool(artifact["valid"])
        validated_artifacts.append(artifact)

    report_path = output_root / "artifact_validation_report.json"
    report_path.write_text(json.dumps({"artifacts": validated_artifacts}, indent=2), encoding="utf-8")
    manifest = ModelValidationManifest(
        run_id=run_context.run_id,
        quant_manifest=str(quant_manifest_path),
        output_root=str(output_root.resolve()),
        validated_artifacts=validated_artifacts,
        valid=valid,
    )
    write_manifest(output_root / "model_validation_manifest.json", manifest)


if __name__ == "__main__":
    main()
