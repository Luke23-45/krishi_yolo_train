"""
Shared helpers for the canonical Hugging Face object-detection dataset.
"""

from __future__ import annotations

import os
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _configure_quiet_hf_io() -> None:
    """
    Disable Hugging Face / tqdm progress bars so large uploads and downloads do
    not spam notebook or terminal output with one line per file.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("DISABLE_TQDM", "1")


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
    _configure_quiet_hf_io()
    try:
        from huggingface_hub import HfApi
        try:
            from huggingface_hub.utils import disable_progress_bars
        except ImportError:
            disable_progress_bars = None
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for packaging and upload. "
            "Install with: pip install huggingface_hub"
        ) from exc
    if disable_progress_bars is not None:
        disable_progress_bars()
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
    curation_summary: Dict | None = None,
) -> str:
    names = schema["names"]
    missing = [names[idx] for idx in range(schema["nc"]) if idx not in class_distribution]
    lines = [
        "---",
        "task_categories:",
        "- object-detection",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: train/metadata.jsonl",
        "  - split: val",
        "    path: val/metadata.jsonl",
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
        "- `train/metadata.jsonl` and `val/metadata.jsonl` are the Hugging Face-friendly metadata files used by the Hub.",
        "- `parquet/train_metadata.parquet` and `parquet/val_metadata.parquet` are compact analytics/export artifacts.",
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
    if curation_summary:
        threshold = curation_summary.get("strict_class_threshold")
        strict_classes = curation_summary.get("strict_classes", [])
        dropped_images = curation_summary.get("dropped_images", 0)
        strict_label = ", ".join(strict_classes) if strict_classes else "none"
        lines.append(
            f"- Adaptive curation was applied with strict mode for classes above {threshold} candidate objects."
        )
        lines.append(f"- Strict classes in this materialization: {strict_label}.")
        lines.append(f"- Images dropped during curation: {dropped_images}.")
    if missing:
        lines.append(f"- Missing classes in this materialization: {', '.join(missing)}")

    return "\n".join(lines) + "\n"


def write_dataset_card(
    root: Path,
    schema: Dict,
    source_stats: Sequence[Dict],
    split_counts: Dict[str, int],
    class_distribution: Dict[int, int],
    curation_summary: Dict | None = None,
) -> Path:
    path = root / "README.md"
    path.write_text(
        build_dataset_card(
            schema,
            source_stats,
            root,
            split_counts,
            class_distribution,
            curation_summary=curation_summary,
        ),
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
                                    ("categories", pa.list_(pa.int32())),
                                    ("category_names", pa.list_(pa.string())),
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


def write_split_metadata_jsonl(rows: Sequence[Dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def read_split_metadata(dataset_root: Path, split: str) -> List[Dict]:
    _, pq = _require_pyarrow()
    path = dataset_root / "parquet" / f"{split}_metadata.parquet"
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
    x1 = max(0.0, min(1.0, x / float(width)))
    y1 = max(0.0, min(1.0, y / float(height)))
    x2 = max(0.0, min(1.0, (x + w) / float(width)))
    y2 = max(0.0, min(1.0, (y + h) / float(height)))

    w_norm = max(0.0, min(1.0, x2 - x1))
    h_norm = max(0.0, min(1.0, y2 - y1))
    x_center = max(0.0, min(1.0, (x1 + x2) / 2.0))
    y_center = max(0.0, min(1.0, (y1 + y2) / 2.0))
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
            rel_path = Path(row["file_name"])
            src_image = dataset_root / split / rel_path
            image_name = rel_path.name
            dst_image = output_root / "images" / split / image_name
            shutil.copy2(src_image, dst_image)

            label_path = output_root / "labels" / split / f"{Path(image_name).stem}.txt"
            objects = row["objects"]
            lines = []
            for category, bbox in zip(objects["categories"], objects["bbox"]):
                lines.append(
                    f"{int(category)} {coco_bbox_to_yolo(bbox, int(row['width']), int(row['height']))}"
                )
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            stats[f"{split}_images"] += 1
            stats["total_labels"] += len(lines)

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
    token = (
        os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HF_TOKEN")
    )
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=private,
        token=token,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_root),
        token=token,
    )
    return f"https://huggingface.co/datasets/{repo_id}"
