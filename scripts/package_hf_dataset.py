"""
Package and optionally upload the canonical Hugging Face dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Package/upload canonical Hugging Face dataset")
    parser.add_argument("--input", type=Path, required=True, help="Canonical HF dataset root")
    parser.add_argument("--repo-id", type=str, help="Hugging Face dataset repo id")
    parser.add_argument(
        "--publish-format",
        choices=("webdataset", "folder"),
        default="webdataset",
        help="Hub publish artifact format",
    )
    parser.add_argument(
        "--publish-output",
        type=Path,
        help="Optional publish artifact directory; defaults next to the input root",
    )
    parser.add_argument(
        "--shard-size-mb",
        type=int,
        default=1024,
        help="Target WebDataset shard size in MB",
    )
    parser.add_argument(
        "--upload-strategy",
        choices=("large-folder", "folder"),
        default="large-folder",
        help="Upload method when publishing to Hugging Face",
    )
    parser.add_argument(
        "--archive-format",
        choices=("zip", "tar.gz"),
        help="Optional convenience archive format",
    )
    parser.add_argument(
        "--archive-output",
        type=Path,
        help="Optional archive output path; defaults next to the input root",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face dataset repo as private if it does not exist",
    )
    args = parser.parse_args()

    dataset_root = args.input.resolve()
    validate_canonical_root(dataset_root)

    publish_root: Path | None = None
    if args.publish_format == "webdataset":
        publish_root = (
            args.publish_output.resolve()
            if args.publish_output
            else dataset_root.parent / f"{dataset_root.name}_webdataset"
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
            "--publish-output is only supported with --publish-format webdataset."
        )
    else:
        publish_root = dataset_root

    if args.archive_format:
        suffix = ".zip" if args.archive_format == "zip" else ".tar.gz"
        archive_output = args.archive_output.resolve() if args.archive_output else dataset_root.with_suffix(suffix)
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
