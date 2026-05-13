from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
    mode: str = "online"


@dataclass
class TrainingConfig:
    # Model selection
    model: str = "yolov8n.pt"
    weights: Optional[str] = None

    # Training hyperparameters
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    patience: int = 15
    device: str = "auto"
    name: Optional[str] = None

    # Data configuration
    data: Optional[str] = None
    manifest: Optional[str] = None

    # Class imbalance mitigation
    balance: bool = False
    rfs_threshold: Optional[float] = None
    beta: float = 0.9999
    no_class_weights: bool = False

    # Optimization
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5

    # Augmentation
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    auto: bool = False

    # Training logic
    close_mosaic: int = 10
    resume_training: bool = False
    fraction: float = 1.0

    # Validation and checkpointing
    val: int = 1
    save_period: int = -1
    cache: str = "false"
    rect: bool = False
    single_cls: bool = False

    # Output configuration
    output_root: Optional[str] = None
    plots: bool = True
    save_json: bool = False
    save_hybrid: bool = False

    # Debugging and development
    dry_run: bool = False
    verbose: bool = True
    seed: int = 0
    deterministic: bool = False
    workers: int = 8
    project: Optional[str] = None

    # Advanced
    overlap_mask: int = 0
    mask_ratio: int = 4
    drop_path: float = 0.0
    dropout: float = 0.0
    freeze: Optional[str] = None


@dataclass
class QuantizationConfig:
    levels: List[str] = field(default_factory=lambda: ["fp32", "fp16", "int8"])
    imgsz: int = 416
    model: Optional[str] = None
    data: Optional[str] = None


@dataclass
class ValidationConfig:
    format: str = "yolo"
    verify_images: bool = False
    dataset: str = "krishi_bouncer_dataset"


@dataclass
class ExportConfig:
    input: str = "hf_dataset"
    output: str = "krishi_bouncer_dataset"


@dataclass
class VisualizationConfig:
    data_yaml: str = "krishi_bouncer_dataset/data.yaml"
    output_dir: str = "outputs/figures"


@dataclass
class DataPackageConfig:
    input: str = "hf_dataset"
    repo_id: Optional[str] = None
    private: bool = False
    upload_strategy: str = "large-folder"
    publish_format: str = "webdataset"
    publish_output: Optional[str] = None
    shard_size_mb: int = 1024
    archive_format: Optional[str] = None
    archive_output: Optional[str] = None


@dataclass
class ModelValidationConfig:
    require_runtime: bool = True


@dataclass
class ModelPackageConfig:
    package_name: str = "krishi_bouncer_model_bundle"
    archive_format: str = "zip"
    include_source_weights: bool = True
    include_reports: bool = True
    repo_id: Optional[str] = None
    private: bool = False


@dataclass
class YoloMLConfig:
    run: RunConfig = field(default_factory=RunConfig)
    materialize: DatasetProvisioningConfig = field(default_factory=DatasetProvisioningConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantize: QuantizationConfig = field(default_factory=QuantizationConfig)
    validate: ValidationConfig = field(default_factory=ValidationConfig)
    export_yolo: ExportConfig = field(default_factory=ExportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    data_package: DataPackageConfig = field(default_factory=DataPackageConfig)
    model_validate: ModelValidationConfig = field(default_factory=ModelValidationConfig)
    package: ModelPackageConfig = field(default_factory=ModelPackageConfig)


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=YoloMLConfig)


def load_config(config_name: str = "config", overrides: Optional[list[str]] = None) -> YoloMLConfig:
    setup_config()
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    merged = OmegaConf.merge(OmegaConf.structured(YoloMLConfig()), cfg)
    resolved = OmegaConf.to_object(merged)
    _validate_telemetry_config(resolved.telemetry)
    return resolved


def _validate_telemetry_config(cfg: TelemetryConfig) -> None:
    valid_modes = {"online", "offline", "disabled"}
    if cfg.mode not in valid_modes:
        raise ValueError(
            f"Invalid telemetry.mode '{cfg.mode}'. Expected one of: {', '.join(sorted(valid_modes))}."
        )


def to_config_dict(cfg: YoloMLConfig | Any) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
