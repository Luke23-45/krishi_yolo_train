from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from hydra.core.config_store import ConfigStore

# ═══════════════════════════════════════════════════════════════════════════════
# DATASET PROVISIONING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetProvisioningConfig:
    local_path: str = "krishi_bouncer_dataset"
    hf_repo_id: str = "hellxhell/krishi-bouncer-dataset"
    cache_dir: Optional[str] = None
    force_download: bool = False
    allow_fallback: bool = True

# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TelemetryConfig:
    project: str = "krishi-vaidya"
    run_name: Optional[str] = None
    enable_wandb: bool = True

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrainingConfig:
    data: str = "krishi_bouncer_dataset/data.yaml"
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

# ═══════════════════════════════════════════════════════════════════════════════
# QUANTIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantizationConfig:
    model: str = "runs/detect/best.pt"
    data: Optional[str] = None
    levels: List[str] = field(default_factory=lambda: ["fp32", "fp16", "int8"])
    imgsz: int = 416
    output: str = "outputs/quantized"

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationConfig:
    dataset: str = "krishi_bouncer_dataset"
    format: str = "yolo"
    verify_images: bool = False

# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExportConfig:
    input: str = "hf_dataset"
    output: str = "krishi_bouncer_dataset"

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisualizationConfig:
    data_yaml: str = "krishi_bouncer_dataset/data.yaml"
    output_dir: str = "outputs/figures"

# ═══════════════════════════════════════════════════════════════════════════════
# PACKAGING CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PackageConfig:
    input: str = "hf_dataset"
    repo_id: Optional[str] = None
    private: bool = False
    upload_strategy: str = "large-folder"
    webdataset: bool = True
    shard_size_mb: int = 1024
    archive_format: Optional[str] = None
    archive_out: Optional[str] = None

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONFIGURATION (Master Schema)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class YoloMLConfig:
    dataset: DatasetProvisioningConfig = field(default_factory=DatasetProvisioningConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    package: PackageConfig = field(default_factory=PackageConfig)


def setup_config() -> None:
    """Register all configurations into the Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=YoloMLConfig)
    cs.store(group="dataset", name="default", node=DatasetProvisioningConfig)
    cs.store(group="telemetry", name="default", node=TelemetryConfig)
    cs.store(group="training", name="default", node=TrainingConfig)
    cs.store(group="quantization", name="default", node=QuantizationConfig)
    cs.store(group="validation", name="default", node=ValidationConfig)
    cs.store(group="export", name="default", node=ExportConfig)
    cs.store(group="visualization", name="default", node=VisualizationConfig)
    cs.store(group="package", name="default", node=PackageConfig)
