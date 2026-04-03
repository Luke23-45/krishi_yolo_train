"""
Export a canonical Hugging Face dataset into Ultralytics YOLO format.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.canonical_dataset import export_yolo_from_canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical dataset to YOLO format")
    parser.add_argument("--input", type=Path, required=True, help="Canonical HF dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Output YOLO dataset root")
    args = parser.parse_args()

    stats = export_yolo_from_canonical(args.input.resolve(), args.output.resolve())
    print(
        f"Exported YOLO dataset to {args.output.resolve()} "
        f"({stats['train_images']} train, {stats['val_images']} val)"
    )


if __name__ == "__main__":
    main()
