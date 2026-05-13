"""
Telemetry Integration for YoloML.
Configures experiment tracking hooks prior to training.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yoloml.config import TelemetryConfig

logger = logging.getLogger("yoloml.telemetry")


def setup_telemetry(cfg: TelemetryConfig) -> None:
    from yoloml.config import _validate_telemetry_config

    _validate_telemetry_config(cfg)

    if not cfg.enable_wandb or cfg.mode == "disabled":
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        os.environ.pop("WANDB_PROJECT", None)
        logger.info("W&B tracking disabled via configuration.")
        return

    os.environ["WANDB_MODE"] = cfg.mode
    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_PROJECT"] = cfg.project

    try:
        import wandb  # noqa: F401
    except ImportError:
        logger.warning(
            "W&B mode is '%s' but the 'wandb' package is not installed. Proceeding without tracking.",
            cfg.mode,
        )
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["WANDB_DISABLED"] = "true"
        os.environ.pop("WANDB_PROJECT", None)
        return

    if cfg.mode == "offline":
        logger.info("Weights & Biases (W&B) offline mode enabled -> Project: %s", cfg.project)
    else:
        logger.info("Weights & Biases (W&B) online mode enabled -> Project: %s", cfg.project)

    if cfg.run_name:
        logger.info("Designated W&B Run Name: %s", cfg.run_name)
