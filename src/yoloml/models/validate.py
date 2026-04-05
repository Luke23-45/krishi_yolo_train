"""
yoloml/models/validate.py
--------------------------
Validate exported model artifacts from the quantization pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from yoloml.config import ModelValidationConfig
from yoloml.models.quantize import validate_tflite
from yoloml.pipeline import (
    ModelValidationManifest,
    QuantManifest,
    create_run_context,
    ensure_stage_dir,
    load_cli_config,
    parse_stage_args,
    read_manifest,
    snapshot_config,
    write_manifest,
)


def main(argv: list[str] | None = None) -> None:
    cli_args, overrides = parse_stage_args("Validate quantized model artifacts", argv=argv)
    cfg = load_cli_config(overrides=overrides)
    if cli_args.run_id:
        cfg.run.run_id = cli_args.run_id
    if cli_args.manifest:
        cfg.model_validation.manifest = cli_args.manifest
    if cli_args.output_root:
        cfg.model_validation.output_root = cli_args.output_root

    args: ModelValidationConfig = cfg.model_validation
    if not args.manifest:
        raise ValueError("Model validation requires --manifest pointing to quant_manifest.json")

    run_context = create_run_context(cfg, run_id=cfg.run.run_id)
    output_root = (
        ensure_stage_dir(Path(args.output_root).resolve())
        if args.output_root
        else ensure_stage_dir(run_context.model_validation_dir)
    )
    snapshot_config(cfg, output_root / "resolved_config.json")

    quant_manifest_path = Path(args.manifest).resolve()
    quant_manifest = read_manifest(quant_manifest_path, QuantManifest)
    if not quant_manifest.valid:
        raise RuntimeError(f"Quantization manifest is not valid: {quant_manifest_path}")

    validated_artifacts = []
    valid = True
    for level in quant_manifest.levels:
        artifact = {
            "level": level.get("level"),
            "success": level.get("success", False),
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
