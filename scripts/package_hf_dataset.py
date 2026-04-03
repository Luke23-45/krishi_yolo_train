"""
Package and optionally upload the canonical Hugging Face dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.canonical_dataset import create_archive, upload_dataset_folder


def validate_canonical_root(dataset_root: Path) -> None:
    required = [
        dataset_root / "classes.json",
        dataset_root / "licenses.json",
        dataset_root / "README.md",
        dataset_root / "train" / "images",
        dataset_root / "train" / "metadata.parquet",
        dataset_root / "val" / "images",
        dataset_root / "val" / "metadata.parquet",
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

    if args.archive_format:
        suffix = ".zip" if args.archive_format == "zip" else ".tar.gz"
        archive_output = args.archive_output.resolve() if args.archive_output else dataset_root.with_suffix(suffix)
        archive_path = create_archive(dataset_root, archive_output, args.archive_format)
        print(f"Created archive: {archive_path}")

    if args.repo_id:
        url = upload_dataset_folder(dataset_root, args.repo_id, private=args.private)
        print(f"Uploaded dataset to {url}")


if __name__ == "__main__":
    main()
