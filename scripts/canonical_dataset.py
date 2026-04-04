"""
Shared helpers for the canonical Hugging Face object-detection dataset.
"""

from __future__ import annotations

import io
import os
import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

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


def build_webdataset_card(
    schema: Dict[int, str] | Dict[str, Any],
    split_counts: Dict[str, int],
    shard_counts: Dict[str, int],
    class_distribution: Dict[int, int],
    source_stats: Sequence[Dict] | None = None,
) -> str:
    names = schema["names"] if isinstance(schema, dict) and "names" in schema else schema
    lines = [
        "---",
        "task_categories:",
        "- object-detection",
        "pretty_name: Krishi Vaidya Bouncer Dataset (WebDataset)",
        "---",
        "",
        "# Krishi Vaidya Bouncer Dataset (WebDataset)",
        "",
        "Hub-native sharded WebDataset export for large-scale object detection training and streaming.",
        "",
        "## Format",
        "",
        "- `train/train-*.tar` and `val/val-*.tar` store sharded samples for streaming-friendly access.",
        "- Each sample is written as `<sample_id>.jpg` plus `<sample_id>.json` inside the TAR shard.",
        "- JSON records keep canonical COCO-style `bbox` annotations in pixel `xywh` form.",
        "- The canonical local dataset and derived YOLO export remain the source-of-truth build artifacts in this repo.",
        "",
        "## Load",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        'ds = load_dataset("webdataset", data_files={"train": "train/*.tar", "val": "val/*.tar"}, split="train")',
        'sample = ds[0]',
        'image = sample[\"jpg\"]',
        'annotation = sample[\"json\"]',
        "```",
        "",
        "## Splits",
        "",
        f"- Train: {split_counts.get('train', 0):,d} samples across {shard_counts.get('train', 0):,d} shards",
        f"- Val: {split_counts.get('val', 0):,d} samples across {shard_counts.get('val', 0):,d} shards",
        "",
        "## Classes",
        "",
    ]
    for idx in sorted(int(k) for k in names):
        lines.append(f"- {idx}: {names[idx] if isinstance(names, dict) else names[int(idx)]}")

    lines.extend(["", "## Distribution", ""])
    for idx in sorted(class_distribution):
        class_name = names[idx] if isinstance(names, dict) else names[int(idx)]
        lines.append(f"- {class_name}: {class_distribution[idx]:,d} objects")

    if source_stats:
        lines.extend(["", "## Sources", ""])
        for src in source_stats:
            lines.append(
                f"- `{src.get('source_name', 'unknown')}`: "
                f"{src.get('source_type', 'unknown')} "
                f"({src.get('source_handle', '')})"
            )

    return "\n".join(lines) + "\n"


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


def _read_licenses(dataset_root: Path) -> List[Dict]:
    licenses_path = dataset_root / "licenses.json"
    if not licenses_path.exists():
        return []
    return json.loads(licenses_path.read_text(encoding="utf-8"))


def _copy_publish_metadata(
    dataset_root: Path,
    output_root: Path,
    readme_text: str,
    manifest: Dict[str, Any],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(readme_text, encoding="utf-8")
    shutil.copy2(dataset_root / "classes.json", output_root / "classes.json")
    if (dataset_root / "licenses.json").exists():
        shutil.copy2(dataset_root / "licenses.json", output_root / "licenses.json")
    if (dataset_root / "materialization_report.json").exists():
        shutil.copy2(
            dataset_root / "materialization_report.json",
            output_root / "materialization_report.json",
        )
    (output_root / "webdataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


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


def export_webdataset_from_canonical(
    dataset_root: Path,
    output_root: Path,
    shard_size_mb: int = 1024,
) -> Dict[str, Any]:
    if dataset_root.resolve() == output_root.resolve():
        raise ValueError("Canonical input and WebDataset output directories must be different.")
    if shard_size_mb <= 0:
        raise ValueError("shard_size_mb must be a positive integer.")

    names = read_schema_names(dataset_root)
    licenses = _read_licenses(dataset_root)
    output_root = output_root.resolve()

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    shard_limit_bytes = shard_size_mb * 1024 * 1024
    split_counts: Dict[str, int] = {}
    shard_counts: Dict[str, int] = {}
    class_distribution: Dict[int, int] = {}
    manifest_shards: List[Dict[str, Any]] = []

    for split in ("train", "val"):
        rows = read_split_metadata(dataset_root, split)
        split_counts[split] = len(rows)
        shard_dir = output_root / split
        shard_dir.mkdir(parents=True, exist_ok=True)

        shard_index = -1
        shard_size = 0
        shard_samples = 0
        tar_handle = None
        tar_path: Path | None = None

        def start_new_shard() -> None:
            nonlocal shard_index, shard_size, shard_samples, tar_handle, tar_path
            if tar_handle is not None:
                tar_handle.close()
                manifest_shards.append(
                    {
                        "split": split,
                        "path": str(tar_path.relative_to(output_root)).replace("\\", "/"),
                        "sample_count": shard_samples,
                        "size_bytes": tar_path.stat().st_size,
                    }
                )
            shard_index += 1
            shard_size = 0
            shard_samples = 0
            tar_path = shard_dir / f"{split}-{shard_index:06d}.tar"
            tar_handle = tarfile.open(tar_path, "w")

        for row in rows:
            if tar_handle is None:
                start_new_shard()

            rel_path = Path(row["file_name"])
            src_image = dataset_root / split / rel_path
            sample_id = str(row["image_id"])
            image_suffix = src_image.suffix.lower() or ".jpg"
            json_payload = {
                "image_id": sample_id,
                "split": split,
                "width": int(row["width"]),
                "height": int(row["height"]),
                "source_name": row.get("source_name", ""),
                "source_type": row.get("source_type", ""),
                "source_handle": row.get("source_handle", ""),
                "file_name": rel_path.name,
                "objects": {
                    "bbox": row["objects"]["bbox"],
                    "categories": [int(v) for v in row["objects"]["categories"]],
                    "category_names": row["objects"]["category_names"],
                    "area": row["objects"]["area"],
                    "iscrowd": row["objects"]["iscrowd"],
                },
            }
            image_size = src_image.stat().st_size
            json_bytes = json.dumps(json_payload, ensure_ascii=False).encode("utf-8")
            projected_size = shard_size + image_size + len(json_bytes)

            if shard_samples > 0 and projected_size > shard_limit_bytes:
                start_new_shard()

            image_member = tarfile.TarInfo(name=f"{sample_id}{image_suffix}")
            image_member.size = image_size
            with open(src_image, "rb") as image_handle:
                tar_handle.addfile(image_member, image_handle)

            json_member = tarfile.TarInfo(name=f"{sample_id}.json")
            json_member.size = len(json_bytes)
            tar_handle.addfile(json_member, io.BytesIO(json_bytes))

            shard_size += image_size + len(json_bytes)
            shard_samples += 1
            for category in json_payload["objects"]["categories"]:
                class_distribution[category] = class_distribution.get(category, 0) + 1

        if tar_handle is not None:
            tar_handle.close()
            manifest_shards.append(
                {
                    "split": split,
                    "path": str(tar_path.relative_to(output_root)).replace("\\", "/"),
                    "sample_count": shard_samples,
                    "size_bytes": tar_path.stat().st_size,
                }
            )
            shard_counts[split] = shard_index + 1
        else:
            shard_counts[split] = 0

    manifest = {
        "format": "webdataset",
        "shard_size_mb": shard_size_mb,
        "split_counts": split_counts,
        "shard_counts": shard_counts,
        "class_distribution": class_distribution,
        "shards": manifest_shards,
        "sources": licenses,
    }
    readme_text = build_webdataset_card(
        {"names": names},
        split_counts,
        shard_counts,
        class_distribution,
        source_stats=licenses,
    )
    _copy_publish_metadata(dataset_root, output_root, readme_text, manifest)
    return manifest


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


def upload_dataset_folder(
    local_root: Path,
    repo_id: str,
    private: bool = False,
    use_large_folder: bool = False,
) -> str:
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
    upload_args = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "folder_path": str(local_root),
        "token": token,
    }
    if use_large_folder:
        if not hasattr(api, "upload_large_folder"):
            raise RuntimeError(
                "huggingface_hub in this environment does not support upload_large_folder(). "
                "Upgrade with: pip install -U huggingface_hub"
            )
        api.upload_large_folder(**upload_args)
    else:
        api.upload_folder(**upload_args)
    return f"https://huggingface.co/datasets/{repo_id}"
