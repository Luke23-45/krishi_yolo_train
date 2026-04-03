"""
Shared helpers for the canonical Hugging Face object-detection dataset.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for canonical dataset metadata. "
            "Install with: pip install pyarrow"
        ) from exc
    return pa, pq


def _require_hf_hub():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for packaging and upload. "
            "Install with: pip install huggingface_hub"
        ) from exc
    return HfApi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_canonical_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)


def write_classes_json(root: Path, schema: Dict) -> Path:
    path = root / "classes.json"
    payload = {
        "nc": int(schema["nc"]),
        "names": {str(k): v for k, v in schema["names"].items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_licenses_json(root: Path, sources: Sequence[Dict]) -> Path:
    path = root / "licenses.json"
    payload = []
    for src in sources:
        payload.append(
            {
                "source_name": src["name"],
                "source_type": src["type"],
                "source_handle": src.get("handle", ""),
                "license": src.get("license", "unknown"),
                "notes": src.get("notes", ""),
            }
        )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_dataset_card(
    schema: Dict,
    source_stats: Sequence[Dict],
    canonical_root: Path,
    split_counts: Dict[str, int],
    class_distribution: Dict[int, int],
) -> str:
    names = schema["names"]
    missing = [names[idx] for idx in range(schema["nc"]) if idx not in class_distribution]
    lines = [
        "---",
        "task_categories:",
        "- object-detection",
        "pretty_name: Krishi Vaidya Bouncer Dataset",
        "---",
        "",
        "# Krishi Vaidya Bouncer Dataset",
        "",
        "Canonical object-detection dataset for Hugging Face, with YOLO exported as a derived training format.",
        "",
        "## Structure",
        "",
        "- `train/images/` and `val/images/` contain the canonical images.",
        "- `train/metadata.parquet` and `val/metadata.parquet` contain per-image metadata and COCO-format bounding boxes.",
        "- `classes.json` stores the canonical class schema.",
        "- `licenses.json` stores source provenance and license notes.",
        "",
        "## Splits",
        "",
        f"- Train: {split_counts.get('train', 0):,d} images",
        f"- Val: {split_counts.get('val', 0):,d} images",
        "",
        "## Classes",
        "",
    ]
    for idx in range(schema["nc"]):
        lines.append(f"- {idx}: {names[idx]}")

    lines.extend(["", "## Sources", ""])
    for src in source_stats:
        lines.append(
            f"- `{src.get('source_name', 'unknown')}`: "
            f"{src.get('source_type', 'unknown')} "
            f"({src.get('source_handle', '')})"
        )

    lines.extend(["", "## Notes", ""])
    lines.append("- Bounding boxes are stored in COCO pixel xywh format.")
    lines.append("- YOLO labels are exported from this canonical dataset, not stored as the master archive.")
    if missing:
        lines.append(f"- Missing classes in this materialization: {', '.join(missing)}")

    return "\n".join(lines) + "\n"


def write_dataset_card(
    root: Path,
    schema: Dict,
    source_stats: Sequence[Dict],
    split_counts: Dict[str, int],
    class_distribution: Dict[int, int],
) -> Path:
    path = root / "README.md"
    path.write_text(
        build_dataset_card(schema, source_stats, root, split_counts, class_distribution),
        encoding="utf-8",
    )
    return path


def _metadata_schema():
    pa, _ = _require_pyarrow()
    return pa.schema(
        [
            ("image_id", pa.string()),
            ("file_name", pa.string()),
            ("width", pa.int32()),
            ("height", pa.int32()),
            ("source_name", pa.string()),
            ("source_type", pa.string()),
            ("source_handle", pa.string()),
            ("split", pa.string()),
            (
                "objects",
                pa.struct(
                    [
                        ("bbox", pa.list_(pa.list_(pa.float32()))),
                        ("category", pa.list_(pa.int32())),
                        ("category_name", pa.list_(pa.string())),
                        ("area", pa.list_(pa.float32())),
                        ("iscrowd", pa.list_(pa.int8())),
                    ]
                ),
            ),
            ("num_objects", pa.int32()),
            ("sha256", pa.string()),
        ]
    )


def write_split_metadata(rows: Sequence[Dict], output_path: Path) -> Path:
    pa, pq = _require_pyarrow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows), schema=_metadata_schema())
    pq.write_table(table, output_path)
    return output_path


def read_split_metadata(dataset_root: Path, split: str) -> List[Dict]:
    _, pq = _require_pyarrow()
    path = dataset_root / split / "metadata.parquet"
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def read_schema_names(dataset_root: Path) -> Dict[int, str]:
    payload = json.loads((dataset_root / "classes.json").read_text(encoding="utf-8"))
    return {int(k): v for k, v in payload["names"].items()}


def write_yolo_data_yaml(output_root: Path, names: Dict[int, str]) -> Path:
    data_yaml_path = output_root / "data.yaml"
    payload = {
        "path": str(output_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    data_yaml_path.write_text(
        yaml.dump(payload, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return data_yaml_path


def coco_bbox_to_yolo(bbox: Sequence[float], width: int, height: int) -> str:
    x, y, w, h = [float(v) for v in bbox]
    x_center = (x + w / 2.0) / float(width)
    y_center = (y + h / 2.0) / float(height)
    w_norm = w / float(width)
    h_norm = h / float(height)
    return f"{x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def export_yolo_from_canonical(dataset_root: Path, output_root: Path) -> Dict[str, int]:
    names = read_schema_names(dataset_root)

    if dataset_root.resolve() == output_root.resolve():
        raise ValueError("Canonical input and YOLO output directories must be different.")

    if output_root.exists():
        shutil.rmtree(output_root)

    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {"train_images": 0, "val_images": 0, "total_labels": 0}

    for split in ("train", "val"):
        rows = read_split_metadata(dataset_root, split)
        for row in rows:
            src_image = dataset_root / split / "images" / row["file_name"]
            dst_image = output_root / "images" / split / row["file_name"]
            shutil.copy2(src_image, dst_image)

            label_path = output_root / "labels" / split / f"{Path(row['file_name']).stem}.txt"
            objects = row["objects"]
            lines = []
            for category, bbox in zip(objects["category"], objects["bbox"]):
                lines.append(
                    f"{int(category)} {coco_bbox_to_yolo(bbox, int(row['width']), int(row['height']))}"
                )
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats[f"{split}_images"] += 1
            stats["total_labels"] += 1

    write_yolo_data_yaml(output_root, names)
    return stats


def create_archive(input_root: Path, archive_path: Path, archive_format: str) -> Path:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_format == "zip":
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in input_root.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(input_root))
        return archive_path

    if archive_format == "tar.gz":
        with tarfile.open(archive_path, "w:gz") as tf:
            tf.add(input_root, arcname=input_root.name)
        return archive_path

    raise ValueError(f"Unsupported archive format: {archive_format}")


def upload_dataset_folder(local_root: Path, repo_id: str, private: bool = False) -> str:
    HfApi = _require_hf_hub()
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_root),
    )
    return f"https://huggingface.co/datasets/{repo_id}"
