from __future__ import annotations

import json
from pathlib import Path

import pytest

from yoloml.data.canonical import (
    ensure_canonical_dirs,
    write_classes_json,
    write_dataset_card,
    write_licenses_json,
    write_split_metadata,
    write_split_metadata_jsonl,
    write_yolo_data_yaml,
)


SCHEMA = {
    "nc": 1,
    "names": {0: "leaf"},
}


def build_yolo_dataset(root: Path) -> Path:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        image_path = root / "images" / split / f"{split}_0.jpg"
        label_path = root / "labels" / split / f"{split}_0.txt"
        image_path.write_bytes(b"fake-image")
        label_path.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    write_yolo_data_yaml(root, SCHEMA["names"])
    return root


def build_canonical_dataset(root: Path) -> Path:
    ensure_canonical_dirs(root)
    write_classes_json(root, SCHEMA)
    write_licenses_json(root, [{"name": "fixture", "type": "local", "handle": str(root)}])

    split_counts = {}
    class_distribution = {0: 2}
    for split in ("train", "val"):
        image_path = root / split / "images" / f"{split}_0.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"fake-image")
        row = {
            "image_id": f"{split}_0",
            "file_name": f"images/{split}_0.jpg",
            "width": 100,
            "height": 100,
            "source_name": "fixture",
            "source_type": "local",
            "source_handle": str(root),
            "split": split,
            "objects": {
                "bbox": [[10.0, 10.0, 20.0, 20.0]],
                "categories": [0],
                "category_names": ["leaf"],
                "area": [400.0],
                "iscrowd": [0],
            },
            "num_objects": 1,
            "sha256": "fixture",
        }
        write_split_metadata([row], root / "parquet" / f"{split}_metadata.parquet")
        write_split_metadata_jsonl([row], root / split / "metadata.jsonl")
        split_counts[split] = 1

    write_dataset_card(root, SCHEMA, [{"source_name": "fixture", "source_type": "local", "source_handle": str(root)}], split_counts, class_distribution)
    (root / "materialization_report.json").write_text(json.dumps({"curation": {}}), encoding="utf-8")
    return root


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> Path:
    return build_yolo_dataset(tmp_path / "yolo")


@pytest.fixture
def canonical_dataset(tmp_path: Path) -> Path:
    return build_canonical_dataset(tmp_path / "canonical")
