from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Type, TypeVar

from yoloml.config import PROJECT_ROOT, YoloMLConfig, load_config, to_config_dict


T = TypeVar("T")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Cannot serialize value of type {type(value)!r}")


@dataclass
class RunContext:
    run_id: str
    run_root: Path
    project_root: Path

    @property
    def data_dir(self) -> Path:
        return self.run_root / "data"

    @property
    def train_dir(self) -> Path:
        return self.run_root / "train"

    @property
    def quantize_dir(self) -> Path:
        return self.run_root / "quantize"

    @property
    def model_validation_dir(self) -> Path:
        return self.run_root / "model_validation"

    @property
    def package_dir(self) -> Path:
        return self.run_root / "package"


@dataclass
class DataManifest:
    run_id: str
    canonical_root: str
    yolo_root: str
    canonical_validation_report: str
    yolo_validation_report: str
    materialization_report: str
    schema_hash: str
    class_names: dict[str, str]
    valid: bool


@dataclass
class TrainManifest:
    run_id: str
    data_manifest: Optional[str]
    verified_data_yaml: str
    training_config_snapshot: str
    output_root: str
    balance_report: str
    telemetry_project: str
    telemetry_run_name: Optional[str]
    best_weights: Optional[str]
    last_weights: Optional[str]
    valid: bool


@dataclass
class QuantManifest:
    run_id: str
    train_manifest: Optional[str]
    source_model: str
    data_yaml: Optional[str]
    output_root: str
    levels: list[dict[str, Any]]
    comparison_report: Optional[str]
    valid: bool


@dataclass
class ModelValidationManifest:
    run_id: str
    quant_manifest: str
    output_root: str
    validated_artifacts: list[dict[str, Any]]
    valid: bool


@dataclass
class PackageManifest:
    run_id: str
    model_validation_manifest: str
    output_root: str
    packaged_bundle: str
    included_artifacts: list[str]
    metadata: dict[str, Any]
    publish_result: Optional[dict[str, Any]]
    valid: bool


def create_run_context(cfg: YoloMLConfig, run_id: Optional[str] = None) -> RunContext:
    actual_run_id = run_id or cfg.run.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = (PROJECT_ROOT / cfg.run.root_dir / actual_run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    return RunContext(run_id=actual_run_id, run_root=run_root, project_root=PROJECT_ROOT)


def ensure_stage_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def write_manifest(path: Path, payload: Any) -> Path:
    return write_json(path, payload)


def read_manifest(path: str | Path, manifest_type: Type[T]) -> T:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    field_names = {item.name for item in fields(manifest_type)}
    filtered = {key: value for key, value in payload.items() if key in field_names}
    return manifest_type(**filtered)


def schema_hash(class_names: dict[int, str] | dict[str, str]) -> str:
    normalized = {str(key): value for key, value in class_names.items()}
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_stage_args(
    description: str,
    argv: Optional[list[str]] = None,
    extra_arguments: Optional[Iterable[tuple[tuple[str, ...], dict[str, Any]]]] = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    if extra_arguments:
        for flags, kwargs in extra_arguments:
            parser.add_argument(*flags, **kwargs)
    args, overrides = parser.parse_known_args(argv)
    return args, overrides


def load_cli_config(config_name: str = "launch", overrides: Optional[list[str]] = None) -> YoloMLConfig:
    return load_config(config_name=config_name, overrides=overrides or [])


def snapshot_config(cfg: YoloMLConfig, path: Path) -> Path:
    return write_json(path, to_config_dict(cfg))
