"""
yoloml/data/package.py
------------------------
Package and optionally upload the canonical Hugging Face dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from yoloml.config import DataPackageConfig
from yoloml.data.canonical import (
    create_archive,
    export_webdataset_from_canonical,
    upload_dataset_folder,
)
from yoloml.pipeline import DataManifest, load_cli_config, parse_stage_args, read_manifest


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


def main(argv: list[str] | None = None) -> None:
    cli_args, overrides = parse_stage_args("Package a canonical dataset", argv=argv)
    cfg = load_cli_config(overrides=overrides)
    args: DataPackageConfig = cfg.data_package

    if cli_args.manifest:
        args.manifest = cli_args.manifest
    if cli_args.output_root:
        args.output_root = cli_args.output_root

    if args.manifest:
        manifest = read_manifest(args.manifest, DataManifest)
        dataset_root = Path(manifest.canonical_root).resolve()
    else:
        dataset_root = Path(args.input).resolve()

    validate_canonical_root(dataset_root)
    output_root = Path(args.output_root).resolve() if args.output_root else dataset_root.parent

    publish_root: Path | None = None
    if args.publish_format == "webdataset":
        publish_root = (
            Path(args.publish_output).resolve()
            if args.publish_output
            else output_root / f"{dataset_root.name}_webdataset"
        )
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
    elif args.publish_output:
        raise ValueError(
            "--publish-output is only supported with publish_format=webdataset."
        )
    else:
        publish_root = dataset_root

    if args.archive_format:
        suffix = ".zip" if args.archive_format == "zip" else ".tar.gz"
        archive_output = (
            Path(args.archive_output).resolve()
            if args.archive_output
            else output_root / f"{dataset_root.name}{suffix}"
        )
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
