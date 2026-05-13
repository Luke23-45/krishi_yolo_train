"""
yoloml/audit.py
----------------
Failure-scan and launch readiness audit for the linked pipeline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from yoloml.config import CONFIG_ROOT, PROJECT_ROOT, YoloMLConfig, load_config, to_config_dict
from yoloml.data.materialize import resolve_output_roots
from yoloml.pipeline import create_run_context


def _can_import(module_name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, str(exc)


def static_contract_audit(cfg: YoloMLConfig) -> dict:
    required_configs = [
        CONFIG_ROOT / "config.yaml",
        CONFIG_ROOT / "run" / "default.yaml",
        CONFIG_ROOT / "dataset" / "default.yaml",
        CONFIG_ROOT / "telemetry" / "default.yaml",
        CONFIG_ROOT / "training" / "default.yaml",
        CONFIG_ROOT / "quantization" / "default.yaml",
        CONFIG_ROOT / "validation" / "default.yaml",
        CONFIG_ROOT / "data_package" / "default.yaml",
        CONFIG_ROOT / "model_validation" / "default.yaml",
        CONFIG_ROOT / "model_package" / "default.yaml",
        CONFIG_ROOT / "sources.yaml",
    ]
    missing_configs = [str(path) for path in required_configs if not path.exists()]

    entrypoints = [
        "yoloml.data.export_yolo",
        "yoloml.data.package",
        "yoloml.data.validate",
        "yoloml.models.quantize",
        "yoloml.models.validate",
        "yoloml.models.package",
        "yoloml.training.train",
        "yoloml.launch",
    ]
    import_failures = {}
    for module_name in entrypoints:
        ok, error = _can_import(module_name)
        if not ok:
            import_failures[module_name] = error

    sources_canonical_root, sources_yolo_root = resolve_output_roots(CONFIG_ROOT / "sources.yaml")
    dataset_alignment = {
        "config_canonical_root": str(Path(cfg.dataset.canonical_root).resolve()),
        "sources_canonical_root": str(sources_canonical_root),
        "config_yolo_root": str(Path(cfg.dataset.yolo_root).resolve()),
        "sources_yolo_root": str(sources_yolo_root),
    }
    roots_aligned = (
        dataset_alignment["config_canonical_root"] == dataset_alignment["sources_canonical_root"]
        and dataset_alignment["config_yolo_root"] == dataset_alignment["sources_yolo_root"]
    )

    return {
        "missing_configs": missing_configs,
        "import_failures": import_failures,
        "config_sections": sorted(to_config_dict(cfg).keys()),
        "dataset_alignment": dataset_alignment,
        "roots_aligned": roots_aligned,
        "valid": not missing_configs and not import_failures and roots_aligned,
    }


def dependency_audit(cfg: YoloMLConfig) -> dict:
    required_modules = {
        "core": ["yaml", "hydra", "omegaconf", "PIL", "pyarrow", "huggingface_hub"],
        "training": ["ultralytics"],
        "model_validation_runtime": ["tensorflow", "tflite_runtime.interpreter"],
    }
    results: dict[str, list[dict[str, str | bool]]] = {}
    valid = True
    for category, modules in required_modules.items():
        results[category] = []
        category_has_option = category == "model_validation_runtime"
        category_ok = False
        for module_name in modules:
            ok, error = _can_import(module_name)
            results[category].append({"module": module_name, "available": ok, "error": error or ""})
            category_ok = category_ok or ok
        if category_has_option:
            valid = valid and category_ok
        else:
            valid = valid and all(item["available"] for item in results[category])

    return {
        "requirements": results,
        "valid": valid,
    }


def dry_run_audit(cfg: YoloMLConfig) -> dict:
    context = create_run_context(cfg, run_id=cfg.run.run_id or "audit")
    sources_canonical_root, sources_yolo_root = resolve_output_roots(CONFIG_ROOT / "sources.yaml")
    paths_are_distinct = len({
        str(context.data_dir),
        str(context.train_dir),
        str(context.quantize_dir),
        str(context.model_validation_dir),
        str(context.package_dir),
    }) == 5
    dataset_paths_are_distinct = str(sources_canonical_root) != str(sources_yolo_root)

    return {
        "run_id": context.run_id,
        "run_root": str(context.run_root),
        "data_dir": str(context.data_dir),
        "train_dir": str(context.train_dir),
        "quantize_dir": str(context.quantize_dir),
        "model_validation_dir": str(context.model_validation_dir),
        "package_dir": str(context.package_dir),
        "sources_canonical_root": str(sources_canonical_root),
        "sources_yolo_root": str(sources_yolo_root),
        "paths_are_distinct": paths_are_distinct,
        "dataset_paths_are_distinct": dataset_paths_are_distinct,
        "valid": paths_are_distinct and dataset_paths_are_distinct,
    }


def smoke_audit(cfg: YoloMLConfig) -> dict:
    fixture_root = PROJECT_ROOT / "tests"
    required = [
        fixture_root / "conftest.py",
        fixture_root / "test_pipeline_wiring.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    return {
        "fixture_root": str(fixture_root),
        "required_files": [str(path) for path in required],
        "missing": missing,
        "note": "This environment still needs a Python runtime to execute the smoke suite.",
        "valid": not missing,
    }


def failure_injection_audit(cfg: YoloMLConfig) -> dict:
    checks = []

    yolo_root = Path(cfg.dataset.yolo_root).resolve()
    canonical_root = Path(cfg.dataset.canonical_root).resolve()
    checks.append({
        "name": "missing_data_yaml",
        "would_fail": not (yolo_root / "data.yaml").exists(),
    })
    checks.append({
        "name": "missing_best_pt_override",
        "would_fail": cfg.quantization.manifest is None and cfg.quantization.model is None,
    })
    checks.append({
        "name": "canonical_only_without_export",
        "would_fail": canonical_root.exists() and not yolo_root.exists(),
    })
    checks.append({
        "name": "missing_hf_token_for_upload",
        "would_fail": (
            (cfg.data_package.repo_id is not None or cfg.model_package.repo_id is not None)
            and os.environ.get("HF_TOKEN") is None
            and os.environ.get("HUGGING_FACE_HUB_TOKEN") is None
        ),
    })
    checks.append({
        "name": "missing_roboflow_key_for_materialization",
        "would_fail": os.environ.get("ROBOFLOW_API_KEY") is None,
    })
    checks.append({
        "name": "missing_wandb_key",
        "would_fail": (
            cfg.telemetry.enable_wandb
            and cfg.telemetry.mode == "online"
            and os.environ.get("WANDB_API_KEY") is None
        ),
    })
    checks.append({
        "name": "runtime_required_for_model_validation",
        "would_fail": cfg.model_validation.require_runtime
        and not (_can_import("tensorflow")[0] or _can_import("tflite_runtime.interpreter")[0]),
    })

    failures = [check for check in checks if check["would_fail"]]
    return {
        "checks": checks,
        "failing_checks": failures,
        "valid": not failures,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run launch readiness audits")
    parser.add_argument("--run-id", type=str, default="audit")
    args, overrides = parser.parse_known_args(argv)
    cfg = load_config(overrides=overrides)
    cfg.run.run_id = args.run_id

    report = {
        "static_contract_audit": static_contract_audit(cfg),
        "dependency_audit": dependency_audit(cfg),
        "dry_run_audit": dry_run_audit(cfg),
        "smoke_audit": smoke_audit(cfg),
        "failure_injection_audit": failure_injection_audit(cfg),
    }
    report["valid"] = all(
        section.get("valid", False)
        for section in report.values()
        if isinstance(section, dict)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
