"""
Package and optionally upload the canonical Hugging Face dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hydra
from yoloml.config import setup_config, YoloMLConfig, PackageConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.canonical_dataset import (
    create_archive,
    export_webdataset_from_canonical,
    upload_dataset_folder,
)


def validate_canonical_root(dataset_root: Path) -> None:
    required = [
        dataset_root / "classes.json",
        dataset_root / "licenses.json",
        dataset_root / "README.md",
        dataset_root / "train" / "images",
        dataset_root / "train" / "metadata.jsonl",
        dataset_root / "val" / "images",
        dataset_root / "val" / "metadata.jsonl",
        dataset_root / "parquet" / "train_metadata.parquet",
        dataset_root / "parquet" / "val_metadata.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Canonical dataset is incomplete. Missing: " + ", ".join(missing)
        )


setup_config()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: YoloMLConfig) -> None:
    args: PackageConfig = cfg.package

    dataset_root = Path(args.input).resolve()
    validate_canonical_root(dataset_root)

    publish_root: Path | None = None
    if args.webdataset:
        publish_root = dataset_root.parent / f"{dataset_root.name}_webdataset"
        manifest = export_webdataset_from_canonical(
            dataset_root,
            publish_root,
            shard_size_mb=args.shard_size_mb,
        )
        print(f"Created WebDataset publish artifact: {publish_root}")
        print(json.dumps({
            "split_counts": manifest["split_counts"],
            "shard_counts": manifest["shard_counts"],
            "shard_size_mb": manifest["shard_size_mb"],
        }, indent=2))
    else:
        publish_root = dataset_root

    if args.archive_format:
        suffix = ".zip" if args.archive_format == "zip" else ".tar.gz"
        archive_output = Path(args.archive_out).resolve() if args.archive_out else dataset_root.with_suffix(suffix)
        archive_path = create_archive(dataset_root, archive_output, args.archive_format)
        print(f"Created archive: {archive_path}")

    if args.repo_id:
        upload_root = publish_root or dataset_root
        url = upload_dataset_folder(
            upload_root,
            args.repo_id,
            private=args.private,
            use_large_folder=(args.upload_strategy == "large-folder"),
        )
        print(f"Uploaded dataset to {url}")


if __name__ == "__main__":
    main()
