"""
Telemetry Integration for YoloML.
Configures experiment tracking hooks prior to training.
"""
import os
import logging
from typing import Optional

from yoloml.config import TelemetryConfig

logger = logging.getLogger("krishi.telemetry")

def setup_telemetry(cfg: TelemetryConfig) -> None:
    """
    Bootstraps the environment for W&B integration.
    """
    if cfg.enable_wandb:
        try:
            import wandb
            # YOLO explicitly checks for WANDB_PROJECT to set the default project.
            os.environ["WANDB_PROJECT"] = cfg.project
            logger.info(f"Weights & Biases (W&B) Enabled -> Project: {cfg.project}")
            
            if cfg.run_name:
                logger.info(f"Designated W&B Run Name: {cfg.run_name}")

        except ImportError:
            logger.warning("W&B is enabled in config, but 'wandb' package is not installed. Proceeding without tracking.")
            os.environ["WANDB_DISABLED"] = "true"
    else:
        logger.info("W&B tracking disabled via configuration.")
        os.environ["WANDB_DISABLED"] = "true"
