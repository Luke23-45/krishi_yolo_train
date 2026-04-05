from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "configs"


@dataclass
class RunConfig:
    run_id: Optional[str] = None
    root_dir: str = "outputs/runs"
    fail_fast: bool = True
    resume: bool = False
    require_valid_previous_stage: bool = True


@dataclass
class DatasetProvisioningConfig:
    yolo_root: str = "krishi_bouncer_dataset"
    canonical_root: str = "hf_dataset"
    hf_repo_id: str = "hellxhell/krishi-bouncer-dataset"
    cache_dir: Optional[str] = None
    force_download: bool = False
    allow_fallback: bool = True
    local_path: Optional[str] = None


@dataclass
class TelemetryConfig:
    project: str = "krishi-vaidya"
    run_name: Optional[str] = None
    enable_wandb: bool = True


@dataclass
class TrainingConfig:
    manifest: Optional[str] = None
    data: Optional[str] = None
    model: str = "yolov8n.pt"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    patience: int = 15
    device: str = "auto"
    name: Optional[str] = None
    balance: bool = False
    rfs_threshold: Optional[float] = None
    beta: float = 0.9999
    no_class_weights: bool = False
    dry_run: bool = False
    output_root: Optional[str] = None


@dataclass
class QuantizationConfig:
    manifest: Optional[str] = None
    model: Optional[str] = None
    data: Optional[str] = None
    levels: List[str] = field(default_factory=lambda: ["fp32", "fp16", "int8"])
    imgsz: int = 416
    output_root: Optional[str] = None


@dataclass
class ValidationConfig:
    manifest: Optional[str] = None
    dataset: str = "krishi_bouncer_dataset"
    format: str = "yolo"
    verify_images: bool = False
    output_root: Optional[str] = None


@dataclass
class ExportConfig:
    manifest: Optional[str] = None
    input: str = "hf_dataset"
    output: str = "krishi_bouncer_dataset"
    output_root: Optional[str] = None


@dataclass
class VisualizationConfig:
    data_yaml: str = "krishi_bouncer_dataset/data.yaml"
    output_dir: str = "outputs/figures"


@dataclass
class DataPackageConfig:
    manifest: Optional[str] = None
    input: str = "hf_dataset"
    repo_id: Optional[str] = None
    private: bool = False
    upload_strategy: str = "large-folder"
    publish_format: str = "webdataset"
    publish_output: Optional[str] = None
    shard_size_mb: int = 1024
    archive_format: Optional[str] = None
    archive_output: Optional[str] = None
    output_root: Optional[str] = None


@dataclass
class ModelValidationConfig:
    manifest: Optional[str] = None
    output_root: Optional[str] = None
    require_runtime: bool = True


@dataclass
class ModelPackageConfig:
    manifest: Optional[str] = None
    output_root: Optional[str] = None
    package_name: str = "krishi_bouncer_model_bundle"
    archive_format: str = "zip"
    include_source_weights: bool = True
    include_reports: bool = True
    repo_id: Optional[str] = None
    private: bool = False


@dataclass
class YoloMLConfig:
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetProvisioningConfig = field(default_factory=DatasetProvisioningConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    data_package: DataPackageConfig = field(default_factory=DataPackageConfig)
    model_validation: ModelValidationConfig = field(default_factory=ModelValidationConfig)
    model_package: ModelPackageConfig = field(default_factory=ModelPackageConfig)


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=YoloMLConfig)
    cs.store(group="run", name="default", node=RunConfig)
    cs.store(group="dataset", name="default", node=DatasetProvisioningConfig)
    cs.store(group="telemetry", name="default", node=TelemetryConfig)
    cs.store(group="training", name="default", node=TrainingConfig)
    cs.store(group="quantization", name="default", node=QuantizationConfig)
    cs.store(group="validation", name="default", node=ValidationConfig)
    cs.store(group="export", name="default", node=ExportConfig)
    cs.store(group="visualization", name="default", node=VisualizationConfig)
    cs.store(group="data_package", name="default", node=DataPackageConfig)
    cs.store(group="model_validation", name="default", node=ModelValidationConfig)
    cs.store(group="model_package", name="default", node=ModelPackageConfig)


def load_config(overrides: Optional[list[str]] = None) -> YoloMLConfig:
    setup_config()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="config", overrides=overrides or [])
    merged = OmegaConf.merge(OmegaConf.structured(YoloMLConfig()), cfg)
    return OmegaConf.to_object(merged)


def to_config_dict(cfg: YoloMLConfig | Any) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
