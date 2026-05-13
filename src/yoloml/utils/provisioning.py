"""
yoloml/utils/provisioning.py
----------------------------
Utility functions for dataset provisioning and synchronization.
Ensures the training environment has the required dataset artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yoloml.config import YoloMLConfig

logger = logging.getLogger("yoloml.provisioning")


def ensure_dataset_ready(cfg: YoloMLConfig) -> Path:
    dataset_cfg = cfg.dataset
    yolo_root = Path(dataset_cfg.yolo_root).resolve()

    is_valid = False
    if yolo_root.exists() and yolo_root.is_dir():
        data_yaml = yolo_root / "data.yaml"
        if data_yaml.exists():
            train_images_dir = yolo_root / "images" / "train"
            if train_images_dir.exists() and any(train_images_dir.iterdir()):
                is_valid = True
                logger.info("Dataset verified at %s", yolo_root)

    if is_valid and not dataset_cfg.force_download:
        return yolo_root / "data.yaml"

    logger.warning("Dataset missing or invalid at %s. Attempting download from Hugging Face.", yolo_root)

    if not dataset_cfg.hf_repo_id:
        raise RuntimeError("hf_repo_id not specified in configuration. Cannot synchronize dataset.")

    try:
        from huggingface_hub import snapshot_download

        logger.info("Downloading dataset repository: %s", dataset_cfg.hf_repo_id)
        snapshot_download(
            repo_id=dataset_cfg.hf_repo_id,
            repo_type="dataset",
            local_dir=yolo_root,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        if (yolo_root / "data.yaml").exists():
            logger.info("Dataset successfully synchronized and verified at %s", yolo_root)
            return yolo_root / "data.yaml"
        else:
            raise RuntimeError("Synchronized content is invalid: data.yaml missing.")

    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub library not found. Please install it to use automatic provisioning."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to synchronize dataset from Hugging Face: {exc}") from exc
