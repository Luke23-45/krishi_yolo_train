"""
DatasetManager: Data Provisioning Architecture
Acts as a DataModule equivalent to ensure idempotency and seamless
dataset synchronization before Ultralytics training begins.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from yoloml.config import DatasetProvisioningConfig
from yoloml.data.canonical import export_yolo_from_canonical

logger = logging.getLogger("krishi.dataset_manager")


@dataclass
class PreparedDataset:
    data_yaml: Path
    yolo_root: Path
    canonical_root: Optional[Path]
    source: str


class DatasetManager:
    """
    Handles resilient dataset provisioning for training.
    """

    def __init__(self, config: DatasetProvisioningConfig):
        self.config = config
        self.yolo_root = Path(self.config.yolo_root or self.config.local_path or "krishi_bouncer_dataset").resolve()
        self.canonical_root = Path(self.config.canonical_root).resolve()
        self.yaml_path = self.yolo_root / "data.yaml"

    def prepare_data(self, output_root: Optional[Path] = None) -> PreparedDataset:
        logger.info("Initializing dataset provisioning protocol...")

        if self.config.force_download and self.yolo_root.exists():
            logger.warning("force_download=True. Purging local YOLO dataset at %s", self.yolo_root)
            shutil.rmtree(self.yolo_root, ignore_errors=True)

        if self._verify_yolo_integrity(self.yolo_root):
            logger.info("Local YOLO dataset HIT -> %s", self.yolo_root)
            return PreparedDataset(
                data_yaml=self._write_resolved_data_yaml(self.yaml_path, output_root),
                yolo_root=self.yolo_root,
                canonical_root=self.canonical_root if self.canonical_root.exists() else None,
                source="local-yolo",
            )

        if self._verify_canonical_integrity(self.canonical_root):
            logger.info("Canonical dataset available locally -> %s", self.canonical_root)
            export_yolo_from_canonical(self.canonical_root, self.yolo_root)
            return PreparedDataset(
                data_yaml=self._write_resolved_data_yaml(self.yaml_path, output_root),
                yolo_root=self.yolo_root,
                canonical_root=self.canonical_root,
                source="local-canonical-export",
            )

        logger.info("Local datasets missing or invalid. Attempting upstream sync.")
        if self.config.hf_repo_id and self._sync_from_huggingface():
            if self._verify_yolo_integrity(self.yolo_root):
                logger.info("Hub sync produced a valid YOLO dataset.")
                return PreparedDataset(
                    data_yaml=self._write_resolved_data_yaml(self.yaml_path, output_root),
                    yolo_root=self.yolo_root,
                    canonical_root=self.canonical_root if self.canonical_root.exists() else None,
                    source="hub-yolo",
                )
            if self._verify_canonical_integrity(self.canonical_root):
                logger.info("Hub sync produced a valid canonical dataset. Exporting YOLO layout.")
                export_yolo_from_canonical(self.canonical_root, self.yolo_root)
                return PreparedDataset(
                    data_yaml=self._write_resolved_data_yaml(self.yaml_path, output_root),
                    yolo_root=self.yolo_root,
                    canonical_root=self.canonical_root,
                    source="hub-canonical-export",
                )

        if self.config.allow_fallback and self._trigger_materialization():
            if self._verify_canonical_integrity(self.canonical_root):
                export_yolo_from_canonical(self.canonical_root, self.yolo_root)
            if self._verify_yolo_integrity(self.yolo_root):
                return PreparedDataset(
                    data_yaml=self._write_resolved_data_yaml(self.yaml_path, output_root),
                    yolo_root=self.yolo_root,
                    canonical_root=self.canonical_root if self.canonical_root.exists() else None,
                    source="materialized",
                )

        logger.critical("Exhausted all dataset provisioning vectors. Training cannot proceed.")
        sys.exit(1)

    def _verify_yolo_integrity(self, directory: Path) -> bool:
        if not directory.exists() or not directory.is_dir():
            return False

        yaml_file = directory / "data.yaml"
        if not yaml_file.exists():
            return False

        try:
            with open(yaml_file, "r", encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}

            for split in ["train", "val"]:
                if split not in cfg:
                    return False
                split_path = directory / cfg[split]
                if not split_path.exists():
                    return False

            return "nc" in cfg and "names" in cfg
        except Exception:
            return False

    def _verify_canonical_integrity(self, directory: Path) -> bool:
        required = [
            directory / "classes.json",
            directory / "train" / "metadata.jsonl",
            directory / "val" / "metadata.jsonl",
            directory / "parquet" / "train_metadata.parquet",
            directory / "parquet" / "val_metadata.parquet",
        ]
        return all(path.exists() for path in required)

    def _write_resolved_data_yaml(self, yaml_file: Path, output_root: Optional[Path]) -> Path:
        with open(yaml_file, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}

        cfg["path"] = str(self.yolo_root)
        target = yaml_file
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)
            target = output_root / "data_resolved.yaml"

        with open(target, "w", encoding="utf-8") as handle:
            yaml.dump(cfg, handle, default_flow_style=False, sort_keys=False)

        return target

    def _sync_from_huggingface(self) -> bool:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            logger.error("huggingface_hub not installed. Cannot sync.")
            return False

        try:
            targets = [self.yolo_root]
            if self.canonical_root != self.yolo_root:
                targets.append(self.canonical_root)

            for target in targets:
                target.mkdir(parents=True, exist_ok=True)
                snapshot_download(
                    repo_id=self.config.hf_repo_id,
                    repo_type="dataset",
                    local_dir=target,
                    local_dir_use_symlinks=False,
                    cache_dir=self.config.cache_dir,
                    resume_download=True,
                )
            return True
        except Exception as exc:
            logger.error("HF sync exception: %s", exc)
            return False

    def _trigger_materialization(self) -> bool:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "yoloml.data.materialize"],
                cwd=str(self.yolo_root.parent),
                check=False,
            )
            return result.returncode == 0
        except Exception as exc:
            logger.error("Materialization logic failed to execute: %s", exc)
            return False
