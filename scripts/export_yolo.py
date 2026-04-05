"""
Export a canonical Hugging Face dataset into Ultralytics YOLO format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hydra
from yoloml.config import setup_config, YoloMLConfig, ExportConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.canonical_dataset import export_yolo_from_canonical


setup_config()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: YoloMLConfig) -> None:
    args: ExportConfig = cfg.export
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    stats = export_yolo_from_canonical(input_path, output_path)
    print(
        f"Exported YOLO dataset to {output_path} "
        f"({stats['train_images']} train, {stats['val_images']} val)"
    )

if __name__ == "__main__":
    main()
