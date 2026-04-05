"""
DatasetManager: Data Provisioning Architecture
Acts as a DataModule equivalent to ensure idempotency and seamless
Hugging Face synchronization before Ultralytics training begins.
"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import yaml

from yoloml.config import DatasetProvisioningConfig

logger = logging.getLogger("krishi.dataset_manager")

class DatasetManager:
    """
    Handles Resilient Dataset Provisioning.
    Implements a strict Execution Matrix:
      1. Local Cache Hit
      2. Hugging Face Hub Synchronization
      3. Dynamic Materialization Fallback
    """

    def __init__(self, config: DatasetProvisioningConfig):
        self.config = config
        self.local_path = Path(self.config.local_path).resolve()
        self.yaml_path = self.local_path / "data.yaml"

    def prepare_data(self) -> Path:
        """
        Executes the provisioning waterfall and returns the absolute path 
        to a verified data.yaml file required by Ultralytics YOLO.
        """
        logger.info("Initializing Dataset Provisioning Protocol...")

        # Condition 0: Force download
        if self.config.force_download and self.local_path.exists():
            logger.warning(f"force_download=True. Purging local cache at {self.local_path}")
            shutil.rmtree(self.local_path, ignore_errors=True)

        # Condition 1: Check Local Cache Hit
        if self._verify_integrity(self.local_path):
            logger.info(f"Local Cache HIT -> {self.local_path}")
            return self._ensure_absolute_paths(self.yaml_path)

        logger.info(f"Local Cache MISS or INVALID at {self.local_path}")

        # Condition 2: Hugging Face Synchronization
        if self.config.hf_repo_id:
            logger.info(f"Attempting upstream pull from Hub: {self.config.hf_repo_id}")
            if self._sync_from_huggingface():
                if self._verify_integrity(self.local_path):
                    logger.info("Hub synchronization successful and verified.")
                    return self._ensure_absolute_paths(self.yaml_path)
                else:
                    logger.error("Hub dataset downloaded but failed schema verification.")
            else:
                logger.warning("Hub synchronization failed or was unavailable.")

        # Condition 3: Dynamic Materialization (Compute-bound Fallback)
        if self.config.allow_fallback:
            logger.info("Firing materialization fallback protocol...")
            if self._trigger_materialization():
                if self._verify_integrity(self.local_path):
                    logger.info("Materialization successful and verified.")
                    return self._ensure_absolute_paths(self.yaml_path)
                else:
                    logger.error("Materialization completed but failed schema verification.")
            else:
                logger.error("Materialization fallback failed.")

        # Terminal Failure
        logger.critical("EXHAUSTED ALL DATA PROVISIONING VECTORS. Training cannot proceed.")
        sys.exit(1)

    def _verify_integrity(self, directory: Path) -> bool:
        """
        Structural verification of a YOLO dataset payload.
        Ensures data.yaml exists and expected splits are resolvable.
        """
        if not directory.exists() or not directory.is_dir():
            return False

        yaml_file = directory / "data.yaml"
        if not yaml_file.exists():
            logger.debug(f"Integrity check failed: {yaml_file} is missing.")
            return False

        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            for split in ["train", "val"]:
                if split not in cfg:
                    logger.debug(f"Integrity check failed: '{split}' key missing in data.yaml.")
                    return False
                
                # Check conceptually if the path resolves
                # (The path in yaml could be relative, YOLO resolves relative to the yaml file)
                split_path = directory / cfg[split]
                if not split_path.parent.exists():
                     logger.debug(f"Integrity check failed: split parent {split_path.parent} missing.")
                     return False
                     
            if "nc" not in cfg or "names" not in cfg:
                logger.debug("Integrity check failed: 'nc' or 'names' missing.")
                return False

            return True

        except Exception as e:
            logger.debug(f"Integrity check exception: {e}")
            return False

    def _ensure_absolute_paths(self, yaml_file: Path) -> Path:
        """
        Ultralytics gracefully accepts paths if the `path` key inside data.yaml
        is set absolutely to the root of the dataset. This overwrites `path:` 
        to ensure it flawlessly works regardless of CWD.
        """
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)

            cfg["path"] = str(self.local_path)

            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
                
            return yaml_file
        except Exception as e:
            logger.error(f"Failed patching data.yaml absolute paths: {e}")
            return yaml_file  # Hope for the best

    def _sync_from_huggingface(self) -> bool:
        """
        Executes atomic synchronization using huggingface_hub.
        Using local_dir downloads straight bypassing deep nested symlinks 
        while preserving internal caching advantages.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error("huggingface_hub not installed. Cannot sync. Run: pip install huggingface_hub")
            return False

        try:
            self.local_path.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=self.config.hf_repo_id,
                repo_type="dataset",
                local_dir=self.local_path,
                local_dir_use_symlinks=False,  # Ensures fully materialized files on Windows
                cache_dir=self.config.cache_dir,
                resume_download=True,
            )
            return True
        except Exception as e:
            logger.error(f"HF sync exception: {e}")
            return False

    def _trigger_materialization(self) -> bool:
        """
        Fallback computing vector. Invokes the materialize script functions natively.
        """
        try:
            # Import natively rather than running subprocess
            from yoloml.data.materialize import main as materialize_main
            logger.info("Executing native materialization protocol...")
            # The materializer itself uses Hydra or config loading. 
            # Because it currently uses pure CLI, wait, does it use Hydra?
            # We didn't migrate materialize_bouncer.py to Hydra in the previous step
            # because the user said it was already config-driven via sources.yaml.
            # So calling main() might parse sys.argv.
            # Instead of fighting sys.argv, we'll try to execute it as a subprocess if needed,
            # but ideally we just invoke its internal logic.
            import subprocess
            import sys
            result = subprocess.run(
                [sys.executable, "-m", "yoloml.data.materialize"],
                cwd=str(self.local_path.parent)
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Materialization logic failed to execute: {e}")
            return False
