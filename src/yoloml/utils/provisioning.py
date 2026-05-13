"""
yoloml/utils/provisioning.py
----------------------------
Utility functions for dataset provisioning and synchronization.
Ensures the training environment has the required dataset artifacts.
"""

from __future__ import annotations

import json
import logging
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yoloml.config import YoloMLConfig

logger = logging.getLogger("yoloml.provisioning")


def _convert_json_to_yolo_txt(json_content: str, txt_path: Path) -> None:
    payload = json.loads(json_content)
    width = int(payload.get("width", 1))
    height = int(payload.get("height", 1))
    objects = payload.get("objects", {})
    
    lines = []
    if "categories" in objects and "bbox" in objects:
        for category, bbox in zip(objects["categories"], objects["bbox"]):
            x, y, w, h = [float(v) for v in bbox]
            x1 = max(0.0, min(1.0, x / float(width)))
            y1 = max(0.0, min(1.0, y / float(height)))
            x2 = max(0.0, min(1.0, (x + w) / float(width)))
            y2 = max(0.0, min(1.0, (y + h) / float(height)))
            w_norm = max(0.0, min(1.0, x2 - x1))
            h_norm = max(0.0, min(1.0, y2 - y1))
            x_center = max(0.0, min(1.0, (x1 + x2) / 2.0))
            y_center = max(0.0, min(1.0, (y1 + y2) / 2.0))
            
            yolo_line = f"{int(category)} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
            lines.append(yolo_line)
            
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unpack_webdataset_tars(yolo_root: Path) -> None:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for split in ["train", "val", "test"]:
        split_dir = yolo_root / split
        if not split_dir.exists():
            continue
            
        # 1. First, handle the case where the user ALREADY extracted the files into `train/` 
        # but they are still `.json` format instead of `.txt`.
        existing_jsons = list(split_dir.glob("*.json"))
        if existing_jsons:
            logger.info(f"Found {len(existing_jsons)} raw WebDataset JSON annotations in '{split}'. Converting to YOLO .txt format...")
            for json_path in existing_jsons:
                txt_path = json_path.with_suffix(".txt")
                if not txt_path.exists():
                    try:
                        _convert_json_to_yolo_txt(json_path.read_text(encoding="utf-8"), txt_path)
                    except Exception as e:
                        logger.warning(f"Failed to convert {json_path.name}: {e}")
        
        # 2. Then, handle tar archives if they exist.
        tars = list(split_dir.glob("*.tar"))
        if not tars:
            continue
            
        images_dir = yolo_root / "images" / split
        labels_dir = yolo_root / "labels" / split
        
        # Check if already unpacked (optimization)
        if images_dir.exists() and any(images_dir.iterdir()):
            continue
            
        logger.info(f"Unpacking {len(tars)} tar archives for '{split}' split...")
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        for tar_path in tars:
            try:
                with tarfile.open(tar_path, "r") as tar:
                    for member in tar.getmembers():
                        if not member.isfile():
                            continue
                        
                        ext = Path(member.name).suffix.lower()
                        if ext in image_extensions:
                            member.name = Path(member.name).name # flatten
                            tar.extract(member, path=images_dir)
                        elif ext == ".json":
                            f = tar.extractfile(member)
                            if f:
                                json_content = f.read().decode("utf-8")
                                txt_filename = Path(member.name).with_suffix(".txt").name
                                txt_path = labels_dir / txt_filename
                                _convert_json_to_yolo_txt(json_content, txt_path)
            except Exception as e:
                logger.warning(f"Failed to unpack {tar_path.name}: {e}")

def ensure_dataset_ready(cfg: YoloMLConfig) -> Path:
    dataset_cfg = cfg.dataset
    yolo_root = Path(dataset_cfg.yolo_root).resolve()

    if yolo_root.exists() and yolo_root.is_dir():
        _unpack_webdataset_tars(yolo_root)

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
            logger.info("Dataset downloaded. Checking for WebDataset tar archives...")
            _unpack_webdataset_tars(yolo_root)
                
            # Verify the structure is correct
            train_images_dir = yolo_root / "images" / "train"
            if train_images_dir.exists() and any(train_images_dir.iterdir()):
                logger.info("Dataset successfully synchronized, unpacked, and verified at %s", yolo_root)
                return yolo_root / "data.yaml"
            else:
                raise RuntimeError("Synchronized content is invalid: images/train directory is missing or empty after unpack.")
        else:
            raise RuntimeError("Synchronized content is invalid: data.yaml missing.")

    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub library not found. Please install it to use automatic provisioning."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to synchronize dataset from Hugging Face: {exc}") from exc
