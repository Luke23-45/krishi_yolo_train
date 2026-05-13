"""
yoloml/launch.py
-----------------
Orchestrate the linked data and model pipelines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from yoloml.config import PROJECT_ROOT, YoloMLConfig, load_config
from yoloml.data.canonical import export_yolo_from_canonical, read_schema_names
from yoloml.data.materialize import resolve_output_roots
from yoloml.data.validate import validate_canonical, validate_yolo
from yoloml.models import package as model_package_module
from yoloml.models import quantize as quantize_module
from yoloml.models import validate as model_validate_module
from yoloml.pipeline import (
    DataManifest,
    ModelValidationManifest,
    PackageManifest,
    QuantManifest,
    TrainManifest,
    create_run_context,
    ensure_stage_dir,
    read_manifest,
    schema_hash,
    snapshot_config,
    write_manifest,
)
from yoloml.data.dataset import DatasetManager
from yoloml.training import train as train_module


def _existing_file(path_value: str | None) -> bool:
    return bool(path_value) and Path(path_value).exists()


def _can_resume_data(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = read_manifest(manifest_path, DataManifest)
    return (
        manifest.valid
        and Path(manifest.canonical_root).exists()
        and Path(manifest.yolo_root).exists()
        and _existing_file(manifest.canonical_validation_report)
        and _existing_file(manifest.yolo_validation_report)
        and _existing_file(manifest.materialization_report)
    )


def _can_resume_train(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = read_manifest(manifest_path, TrainManifest)
    return (
        manifest.valid
        and _existing_file(manifest.verified_data_yaml)
        and _existing_file(manifest.training_config_snapshot)
        and _existing_file(manifest.balance_report)
        and (_existing_file(manifest.best_weights) or _existing_file(manifest.last_weights))
    )


def _can_resume_quant(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = read_manifest(manifest_path, QuantManifest)
    return manifest.valid and all(
        level.get("success") and _existing_file(level.get("tflite_path")) and _existing_file(level.get("metadata_path"))
        for level in manifest.levels
    )


def _can_resume_model_validation(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = read_manifest(manifest_path, ModelValidationManifest)
    return manifest.valid and all(
        artifact.get("valid") and _existing_file(artifact.get("tflite_path"))
        for artifact in manifest.validated_artifacts
    )


def _can_resume_package(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    manifest = read_manifest(manifest_path, PackageManifest)
    return manifest.valid and _existing_file(manifest.packaged_bundle)


def _load_cfg_from_cli(argv: list[str] | None = None) -> tuple[YoloMLConfig, list[str]]:
    # Simply load config with overrides. Hydra will handle any key=value pairs.
    cfg = load_config(overrides=argv)
    return cfg, argv


def run_data_pipeline(cfg: YoloMLConfig, run_id: str) -> Path:
    run_context = create_run_context(cfg, run_id=run_id)
    output_root = ensure_stage_dir(run_context.data_dir)
    manifest_path = output_root / "data_manifest.json"

    if cfg.run.resume and _can_resume_data(manifest_path):
        return manifest_path

    sources_config = PROJECT_ROOT / "configs" / "sources.yaml"
    canonical_root, yolo_root = resolve_output_roots(sources_config)

    # Keep the runtime config aligned with the materialization source of truth.
    cfg.dataset.canonical_root = str(canonical_root)
    cfg.dataset.yolo_root = str(yolo_root)
    snapshot_config(cfg, output_root / "resolved_config.json")

    manager = DatasetManager(cfg.dataset)
    prepared = manager.prepare_data(output_root=output_root)
    canonical_root = Path(prepared.canonical_root).resolve() if prepared.canonical_root else canonical_root
    yolo_root = Path(prepared.yolo_root).resolve()

    if not canonical_root.exists():
        raise FileNotFoundError(
            f"Canonical dataset root not found after provisioning: {canonical_root}. "
            "The launch data stage requires a canonical dataset for validation."
        )

    if not (yolo_root / "data.yaml").exists():
        export_yolo_from_canonical(canonical_root, yolo_root)

    canonical_validation_report = validate_canonical(
        canonical_root,
        do_check_images=cfg.validation.verify_images,
        report_path=output_root / "canonical_validation_report.json",
    )
    yolo_validation_report = validate_yolo(
        yolo_root,
        do_check_images=cfg.validation.verify_images,
        report_path=output_root / "yolo_validation_report.json",
    )

    materialization_report = canonical_root / "materialization_report.json"
    if not materialization_report.exists():
        materialization_report.write_text("{}", encoding="utf-8")
    names = read_schema_names(canonical_root)
    manifest = DataManifest(
        run_id=run_id,
        canonical_root=str(canonical_root),
        yolo_root=str(yolo_root),
        canonical_validation_report=str(canonical_validation_report),
        yolo_validation_report=str(yolo_validation_report),
        materialization_report=str(materialization_report.resolve()),
        schema_hash=schema_hash(names),
        class_names={str(key): value for key, value in names.items()},
        valid=True,
    )
    write_manifest(manifest_path, manifest)
    return manifest_path


def run_model_pipeline(cfg: YoloMLConfig, run_id: str, data_manifest_path: Path) -> dict[str, Path]:
    run_context = create_run_context(cfg, run_id=run_id)

    train_output = ensure_stage_dir(run_context.train_dir)
    quant_output = ensure_stage_dir(run_context.quantize_dir)
    model_validation_output = ensure_stage_dir(run_context.model_validation_dir)
    package_output = ensure_stage_dir(run_context.package_dir)

    train_manifest_path = train_output / "train_manifest.json"
    quant_manifest_path = quant_output / "quant_manifest.json"
    model_validation_manifest_path = model_validation_output / "model_validation_manifest.json"
    package_manifest_path = package_output / "package_manifest.json"

    if not (cfg.run.resume and _can_resume_train(train_manifest_path)):
        train_module.main([
            f"training.manifest={data_manifest_path}",
            f"run.run_id={run_id}",
            f"training.output_root={train_output}",
        ])

    if cfg.run.require_valid_previous_stage and not read_manifest(train_manifest_path, TrainManifest).valid:
        raise RuntimeError(f"Train manifest is not valid: {train_manifest_path}")

    if not (cfg.run.resume and _can_resume_quant(quant_manifest_path)):
        quantize_module.main([
            f"quantization.manifest={train_manifest_path}",
            f"run.run_id={run_id}",
            f"quantization.output_root={quant_output}",
        ])

    if cfg.run.require_valid_previous_stage and not read_manifest(quant_manifest_path, QuantManifest).valid:
        raise RuntimeError(f"Quantization manifest is not valid: {quant_manifest_path}")

    if not (cfg.run.resume and _can_resume_model_validation(model_validation_manifest_path)):
        model_validate_module.main([
            f"model_validation.manifest={quant_manifest_path}",
            f"run.run_id={run_id}",
            f"model_validation.output_root={model_validation_output}",
        ])

    if cfg.run.require_valid_previous_stage and not read_manifest(model_validation_manifest_path, ModelValidationManifest).valid:
        raise RuntimeError(f"Model validation manifest is not valid: {model_validation_manifest_path}")

    if not (cfg.run.resume and _can_resume_package(package_manifest_path)):
        model_package_module.main([
            f"model_package.manifest={model_validation_manifest_path}",
            f"run.run_id={run_id}",
            f"model_package.output_root={package_output}",
        ])

    if cfg.run.require_valid_previous_stage and not read_manifest(package_manifest_path, PackageManifest).valid:
        raise RuntimeError(f"Package manifest is not valid: {package_manifest_path}")

    return {
        "train_manifest": train_manifest_path,
        "quant_manifest": quant_manifest_path,
        "model_validation_manifest": model_validation_manifest_path,
        "package_manifest": package_manifest_path,
    }


def main(argv: list[str] | None = None) -> None:
    cfg, _ = _load_cfg_from_cli(argv)
    run_context = create_run_context(cfg, run_id=cfg.run.run_id)
    
    # We rely on cfg.training.manifest if set, or look for it in the default data dir.
    data_manifest_path = Path(cfg.training.manifest).resolve() if cfg.training.manifest else run_context.data_dir / "data_manifest.json"

    # Launch mode is now controlled via config overrides if needed, e.g. launch.mode=data-only
    # or just by calling specific parts. For simplicity, we keep the default "full" logic
    # but based on config fields if we added them. For now, we assume 'full' or use simple logic.
    
    data_manifest_path = run_data_pipeline(cfg, run_context.run_id)
    print(f"Data pipeline manifest: {data_manifest_path}")

    if not data_manifest_path.exists():
        raise FileNotFoundError(
            f"Data manifest not found: {data_manifest_path}."
        )
    outputs = run_model_pipeline(cfg, run_context.run_id, data_manifest_path)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
