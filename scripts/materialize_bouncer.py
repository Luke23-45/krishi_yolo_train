"""
Canonical-first materialization pipeline for the Krishi Vaidya dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import urlopen

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapters import get_adapter  # noqa: E402
from scripts.canonical_dataset import (  # noqa: E402
    export_yolo_from_canonical,
    ensure_canonical_dirs,
    sha256_file,
    write_classes_json,
    write_dataset_card,
    write_licenses_json,
    write_split_metadata,
    write_split_metadata_jsonl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishi.materialize")


DEFAULT_CURATION = {
    "enabled": True,
    "strict_class_threshold": 2000,
    "strict_drop_policy": "image",
    "min_bbox_area_pixels": 16.0,
    "min_bbox_area_ratio": 0.0001,
    "report_top_failures": 20,
}


def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    required = ("schema", "output", "sources")
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: '{key}'")

    schema = cfg["schema"]
    if "nc" not in schema or "names" not in schema:
        raise ValueError("Config 'schema' must contain 'nc' and 'names'")
    if len(schema["names"]) != schema["nc"]:
        raise ValueError(
            f"Schema declares nc={schema['nc']} but names has {len(schema['names'])} entries"
        )
    return cfg


def resolve_configured_path(path_value: str | Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if expanded.is_absolute():
        return expanded
    return (PROJECT_ROOT / expanded).resolve()


def load_curation_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    curation = dict(DEFAULT_CURATION)
    curation.update(cfg.get("curation", {}) or {})
    curation["enabled"] = bool(curation.get("enabled", True))
    curation["strict_class_threshold"] = int(curation.get("strict_class_threshold", 2000))
    curation["min_bbox_area_pixels"] = float(curation.get("min_bbox_area_pixels", 16.0))
    curation["min_bbox_area_ratio"] = float(curation.get("min_bbox_area_ratio", 0.0001))
    curation["report_top_failures"] = int(curation.get("report_top_failures", 20))
    policy = str(curation.get("strict_drop_policy", "image")).strip().lower()
    if policy != "image":
        raise ValueError("Only strict_drop_policy=image is supported in this version.")
    curation["strict_drop_policy"] = policy
    return curation


def check_canonical_integrity(output_dir: Path, schema: Dict) -> bool:
    classes_path = output_dir / "classes.json"
    if not classes_path.exists():
        return False

    try:
        payload = json.loads(classes_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    names = {int(k): v for k, v in payload.get("names", {}).items()}
    if payload.get("nc") != schema["nc"] or names != schema["names"]:
        return False

    for split in ("train", "val"):
        if not (output_dir / split / "images").exists():
            return False
        if not (output_dir / split / "metadata.jsonl").exists():
            return False
        if not (output_dir / "parquet" / f"{split}_metadata.parquet").exists():
            return False
    return True


def fetch_kaggle(handle: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for Kaggle downloads. Install with: pip install kagglehub"
        ) from exc

    logger.info("  Downloading from Kaggle: %s", handle)
    start = time.time()
    try:
        local_path = kagglehub.dataset_download(handle)
    except Exception as exc:
        raise RuntimeError(f"Kaggle download failed for '{handle}': {exc}") from exc
    logger.info("  Downloaded in %.1fs -> %s", time.time() - start, local_path)
    return Path(local_path)


def _extract_version_numbers(versions_obj: Any) -> List[int]:
    """Extract numeric version ids from SDK or REST response payloads."""
    candidates: List[int] = []

    def add_from_value(value: Any) -> None:
        if isinstance(value, int):
            candidates.append(value)
            return
        if isinstance(value, str):
            tail = value.rsplit("/", 1)[-1]
            if tail.isdigit():
                candidates.append(int(tail))
            return
        if isinstance(value, dict):
            matched = False
            for key in ("version", "id"):
                if key in value:
                    matched = True
                    add_from_value(value[key])
            if not matched:
                for nested_value in value.values():
                    add_from_value(nested_value)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add_from_value(item)
            return
        if hasattr(value, "values") and callable(value.values):
            add_from_value(list(value.values()))

    add_from_value(versions_obj)
    return sorted(set(candidates))


def _resolve_latest_roboflow_version_sdk(project: Any) -> Optional[int]:
    try:
        versions_obj = project.versions()
    except Exception:
        return None

    candidates = _extract_version_numbers(versions_obj)
    return max(candidates) if candidates else None


def _resolve_latest_roboflow_version_api(
    workspace: str,
    project_name: str,
    api_key: str,
) -> Optional[int]:
    """
    Resolve the newest project version via Roboflow's documented project endpoint.
    Docs: https://docs.roboflow.com/developer/rest-api/get-a-project-and-list-versions
    """
    query = urlencode({"api_key": api_key})
    url = f"https://api.roboflow.com/{workspace}/{project_name}?{query}"
    try:
        with urlopen(url, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    project_payload = payload.get("project", {})
    candidates = _extract_version_numbers(project_payload.get("versions", []))
    return max(candidates) if candidates else None


def fetch_roboflow(handle: str, cache_dir: Path) -> Path:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY environment variable is required for Roboflow downloads.")

    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError("roboflow is required. Install with: pip install roboflow") from exc

    parts = handle.split("/")
    if len(parts) != 3:
        raise ValueError(f"Roboflow handle must be 'workspace/project/version', got: '{handle}'")

    workspace, project_name, version_num = parts
    output_dir = cache_dir / "roboflow"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("  Fetching Roboflow dataset: %s", handle)
    start = time.time()
    original_cwd = os.getcwd()

    try:
        os.chdir(output_dir)
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        requested_version = int(version_num)
        try:
            dataset = project.version(requested_version).download("yolov8")
        except Exception as exc:
            fallback_version = (
                _resolve_latest_roboflow_version_sdk(project)
                or _resolve_latest_roboflow_version_api(workspace, project_name, api_key)
            )
            if fallback_version is None or fallback_version == requested_version:
                raise exc
            logger.warning(
                "  Requested Roboflow version %s for %s/%s is unavailable; retrying latest available version %s",
                requested_version,
                workspace,
                project_name,
                fallback_version,
            )
            dataset = project.version(fallback_version).download("yolov8")
        final_path = Path(dataset.location)
    except Exception as exc:
        raise RuntimeError(f"Roboflow download failed for '{handle}': {exc}") from exc
    finally:
        os.chdir(original_cwd)

    logger.info("  Fetched in %.1fs -> %s", time.time() - start, final_path)
    return final_path


def validate_local_path(path_str: str, source_name: str) -> Path:
    path = resolve_configured_path(path_str)
    if path_str == "REPLACE_WITH_LOCAL_PATH":
        raise ValueError(
            f"Source '{source_name}' has a placeholder path. Either provide the real path in sources.yaml or disable it."
        )
    if not path.exists():
        raise FileNotFoundError(f"Source '{source_name}' path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Source '{source_name}' path is not a directory: {path}")
    return path


def _evaluate_object_quality(obj: Any, sample: Any, curation_cfg: Dict[str, Any]) -> List[str]:
    reasons = list(dict.fromkeys(getattr(obj, "quality_flags", []) or []))
    bbox = getattr(obj, "bbox", []) or []
    raw_bbox = getattr(obj, "raw_bbox", []) or []

    if len(raw_bbox) == 4 and any(not math.isfinite(float(v)) for v in raw_bbox):
        reasons.append("nan")

    if not getattr(obj, "valid_geometry", False) or len(bbox) != 4:
        if "collapsed_after_clamp" not in reasons and "malformed" not in reasons:
            reasons.append("invalid_geometry")
        return list(dict.fromkeys(reasons))

    area = float(bbox[2]) * float(bbox[3])
    image_area = max(1.0, float(sample.width) * float(sample.height))

    if area < float(curation_cfg["min_bbox_area_pixels"]):
        reasons.append("too_small_pixels")
    if (area / image_area) < float(curation_cfg["min_bbox_area_ratio"]):
        reasons.append("too_small_ratio")

    return list(dict.fromkeys(reasons))


def curate_samples(
    samples: List[Any],
    schema: Dict,
    curation_cfg: Dict[str, Any],
) -> tuple[List[Any], Dict[str, Any]]:
    candidate_object_counts: Counter = Counter()
    for sample in samples:
        for obj in sample.objects:
            candidate_object_counts[int(obj.category)] += 1

    strict_classes = {
        class_id
        for class_id, count in candidate_object_counts.items()
        if count > curation_cfg["strict_class_threshold"]
    }

    curated_samples: List[Any] = []
    dropped_images_by_class: Counter = Counter()
    dropped_images_by_source: Counter = Counter()
    dropped_objects_by_reason: Counter = Counter()
    dropped_objects_by_class: Counter = Counter()
    dropped_objects_total = 0
    dropped_images_total = 0
    failure_examples: List[Dict[str, Any]] = []

    for sample in samples:
        keep_objects = []
        strict_failure_reasons: Dict[int, List[str]] = {}

        for obj in sample.objects:
            reasons = _evaluate_object_quality(obj, sample, curation_cfg)
            is_strict_class = int(obj.category) in strict_classes

            if is_strict_class and reasons:
                strict_failure_reasons.setdefault(int(obj.category), []).extend(reasons)
                continue

            permissive_reasons = [
                reason
                for reason in reasons
                if reason not in {"out_of_bounds", "truncated_after_clamp"}
            ]
            if permissive_reasons:
                dropped_objects_total += 1
                dropped_objects_by_class[int(obj.category)] += 1
                for reason in permissive_reasons:
                    dropped_objects_by_reason[reason] += 1
                if len(failure_examples) < curation_cfg["report_top_failures"]:
                    failure_examples.append(
                        {
                            "source_name": sample.source_name,
                            "image_id": sample.image_id,
                            "class_name": schema["names"][int(obj.category)],
                            "mode": "permissive-object-drop",
                            "reasons": permissive_reasons,
                        }
                    )
                continue

            keep_objects.append(obj)

        if strict_failure_reasons:
            dropped_images_total += 1
            dropped_images_by_source[sample.source_name] += 1
            for class_id, reasons in strict_failure_reasons.items():
                dropped_images_by_class[class_id] += 1
                for reason in reasons:
                    dropped_objects_by_reason[reason] += 1
                    dropped_objects_total += 1
                if len(failure_examples) < curation_cfg["report_top_failures"]:
                    failure_examples.append(
                        {
                            "source_name": sample.source_name,
                            "image_id": sample.image_id,
                            "class_name": schema["names"][class_id],
                            "mode": "strict-image-drop",
                            "reasons": list(dict.fromkeys(reasons)),
                        }
                    )
            continue

        if not keep_objects:
            dropped_images_total += 1
            dropped_images_by_source[sample.source_name] += 1
            continue

        sample.objects = keep_objects
        curated_samples.append(sample)

    final_distribution: Counter = Counter()
    for sample in curated_samples:
        for obj in sample.objects:
            final_distribution[int(obj.category)] += 1

    summary = {
        "enabled": curation_cfg["enabled"],
        "strict_class_threshold": curation_cfg["strict_class_threshold"],
        "strict_drop_policy": curation_cfg["strict_drop_policy"],
        "min_bbox_area_pixels": curation_cfg["min_bbox_area_pixels"],
        "min_bbox_area_ratio": curation_cfg["min_bbox_area_ratio"],
        "candidate_images": len(samples),
        "candidate_objects": int(sum(candidate_object_counts.values())),
        "candidate_class_distribution": dict(sorted(candidate_object_counts.items())),
        "strict_classes": [schema["names"][class_id] for class_id in sorted(strict_classes)],
        "strict_class_ids": sorted(strict_classes),
        "dropped_images": dropped_images_total,
        "dropped_images_by_class": {
            schema["names"][class_id]: count
            for class_id, count in sorted(dropped_images_by_class.items())
        },
        "dropped_images_by_source": dict(sorted(dropped_images_by_source.items())),
        "dropped_objects": dropped_objects_total,
        "dropped_objects_by_reason": dict(sorted(dropped_objects_by_reason.items())),
        "dropped_objects_by_class": {
            schema["names"][class_id]: count
            for class_id, count in sorted(dropped_objects_by_class.items())
        },
        "final_images": len(curated_samples),
        "final_objects": int(sum(final_distribution.values())),
        "final_class_distribution": dict(sorted(final_distribution.items())),
        "failure_examples": failure_examples,
    }
    return curated_samples, summary


def _materialize_canonical_dataset(
    canonical_root: Path,
    samples: List[Any],
    schema: Dict,
    source_stats: List[Dict],
    sources_cfg: List[Dict],
    curation_summary: Dict[str, Any],
) -> Dict[str, Any]:
    if canonical_root.exists():
        import shutil

        shutil.rmtree(canonical_root)

    ensure_canonical_dirs(canonical_root)
    split_rows: Dict[str, List[Dict]] = {"train": [], "val": []}
    split_counts = {"train": 0, "val": 0}
    class_distribution: Dict[int, int] = defaultdict(int)

    import shutil

    for sample in samples:
        dst_image = canonical_root / sample.split / "images" / sample.output_file_name
        shutil.copy2(sample.image_path, dst_image)
        checksum = sha256_file(dst_image)

        for obj in sample.objects:
            if not obj.category_name:
                obj.category_name = schema["names"][obj.category]
            class_distribution[obj.category] += 1

        split_rows[sample.split].append(sample.to_metadata_row(checksum))
        split_counts[sample.split] += 1

    for split in ("train", "val"):
        write_split_metadata(split_rows[split], canonical_root / "parquet" / f"{split}_metadata.parquet")
        write_split_metadata_jsonl(split_rows[split], canonical_root / split / "metadata.jsonl")

    write_classes_json(canonical_root, schema)
    write_licenses_json(canonical_root, sources_cfg)
    write_dataset_card(
        canonical_root,
        schema,
        source_stats,
        split_counts,
        dict(class_distribution),
        curation_summary=curation_summary,
    )

    return {
        "train_images": split_counts["train"],
        "val_images": split_counts["val"],
        "total_images": split_counts["train"] + split_counts["val"],
        "class_distribution": dict(sorted(class_distribution.items())),
    }


def _sanitize_source_stats(source_stats: List[Dict]) -> List[Dict]:
    return [{k: v for k, v in item.items() if k != "samples"} for item in source_stats]


def generate_report(
    canonical_root: Path,
    source_stats: List[Dict],
    canonical_stats: Dict[str, Any],
    yolo_stats: Dict[str, Any],
    curation_stats: Dict[str, Any],
    elapsed: float,
) -> Path:
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "canonical_output_dir": str(canonical_root.resolve()),
        "elapsed_seconds": round(elapsed, 1),
        "sources": _sanitize_source_stats(source_stats),
        "curation": curation_stats,
        "canonical": canonical_stats,
        "yolo": yolo_stats,
    }
    total_images = sum(item.get("images_processed", 0) for item in source_stats)
    total_errors = sum(item.get("errors", 0) for item in source_stats)
    unmapped = sorted({name for item in source_stats for name in item.get("unmapped_classes", [])})
    report["totals"] = {
        "images_processed": total_images,
        "errors": total_errors,
        "unmapped_classes": unmapped,
    }

    report_path = canonical_root / "materialization_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def materialize(
    config_path: Path,
    force: bool = False,
    dry_run: bool = False,
    output_format: str = "both",
) -> None:
    banner = """
    ============================================================
      KRISHI VAIDYA - Canonical Dataset Materialization v2.0
    ============================================================
    """
    logger.info(banner)

    start_time = time.time()
    cfg = load_config(config_path)
    schema = cfg["schema"]
    output_cfg = cfg["output"]
    curation_cfg = load_curation_config(cfg)
    sources = cfg["sources"]

    canonical_root = resolve_configured_path(output_cfg.get("canonical_root", "hf_dataset"))
    yolo_root = resolve_configured_path(output_cfg.get("yolo_root", output_cfg["root"]))
    cache_dir = PROJECT_ROOT / ".cache"
    val_ratio = output_cfg.get("val_split_ratio", 0.2)
    enabled = [src for src in sources if src.get("enabled", True)]

    logger.info("Config loaded: %s/%s sources enabled", len(enabled), len(sources))
    logger.info("Schema: %s classes", schema["nc"])
    logger.info(
        "Adaptive curation: %s (strict threshold=%s)",
        "enabled" if curation_cfg["enabled"] else "disabled",
        curation_cfg["strict_class_threshold"],
    )
    logger.info("Canonical output: %s", canonical_root)
    logger.info("YOLO output: %s", yolo_root)
    logger.info("Output format: %s", output_format)

    if dry_run:
        logger.info("-" * 60)
        logger.info("DRY RUN - validating config only, no downloads or writes")
        logger.info("-" * 60)
        for src in enabled:
            logger.info("  [%s] %s - %s", src["type"].upper(), src["name"], src.get("handle", "N/A"))
        logger.info("Config is valid. Exiting dry-run.")
        return

    if not force and check_canonical_integrity(canonical_root, schema):
        logger.info("-" * 60)
        logger.info("Valid canonical dataset found locally. Skipping materialization.")
        logger.info("Use --force to rebuild.")
        logger.info("-" * 60)
        if output_format in {"yolo", "both"} and not yolo_root.exists():
            export_yolo_from_canonical(canonical_root, yolo_root)
        return

    source_stats: List[Dict] = []
    all_samples: List[Any] = []

    for idx, src_cfg in enumerate(enabled, start=1):
        name = src_cfg["name"]
        src_type = src_cfg["type"]
        handle = src_cfg.get("handle", "")
        fmt = src_cfg["format"]
        split_strategy = src_cfg.get("split", "preserve")
        class_map = src_cfg.get("class_map", {})

        logger.info("-" * 60)
        logger.info("SOURCE %s/%s | %s", idx, len(enabled), name)
        logger.info("  Type: %s | Format: %s | Split: %s", src_type, fmt, split_strategy)
        logger.info("-" * 60)

        try:
            if src_type == "kaggle":
                source_dir = fetch_kaggle(handle)
            elif src_type == "roboflow":
                source_dir = fetch_roboflow(handle, cache_dir)
            elif src_type == "local":
                source_dir = validate_local_path(handle, name)
            else:
                raise ValueError(f"Unknown source type: '{src_type}'")

            adapter = get_adapter(fmt)()
            stats = adapter.process(
                source_dir=source_dir,
                staging_dir=PROJECT_ROOT / ".staging",
                class_map=class_map,
                source_name=name,
                split_strategy=split_strategy,
                val_ratio=val_ratio,
            )
            for sample in stats.get("samples", []):
                sample.source_type = src_type
                sample.source_handle = handle

            stats["source_name"] = name
            stats["source_type"] = src_type
            stats["source_handle"] = handle
            source_stats.append(stats)
            all_samples.extend(stats.get("samples", []))
        except Exception as exc:
            logger.error("  FAILED: %s", exc)
            source_stats.append(
                {
                    "source_name": name,
                    "source_type": src_type,
                    "source_handle": handle,
                    "images_processed": 0,
                    "errors": 1,
                    "error_message": str(exc),
                    "samples": [],
                }
            )

    total_processed = sum(item.get("images_processed", 0) for item in source_stats)
    if total_processed == 0 or not all_samples:
        logger.error("=" * 60)
        logger.error("FATAL: No images were processed from any source.")
        logger.error("Check your sources.yaml and source credentials.")
        logger.error("=" * 60)
        sys.exit(1)

    curation_stats: Dict[str, Any] = {}
    if curation_cfg["enabled"]:
        logger.info("-" * 60)
        logger.info("CURATING SAMPLES")
        logger.info("-" * 60)
        all_samples, curation_stats = curate_samples(all_samples, schema, curation_cfg)
        logger.info(
            "  Curation kept %s/%s images and %s/%s objects",
            curation_stats.get("final_images", 0),
            curation_stats.get("candidate_images", 0),
            curation_stats.get("final_objects", 0),
            curation_stats.get("candidate_objects", 0),
        )
        if curation_stats.get("strict_classes"):
            logger.info("  Strict classes: %s", curation_stats["strict_classes"])
        if curation_stats.get("dropped_images"):
            logger.info("  Dropped images: %s", curation_stats["dropped_images"])
        if not all_samples:
            logger.error("FATAL: Curation removed every sample. Adjust the thresholds or source config.")
            sys.exit(1)

    canonical_stats: Dict[str, Any] = {}
    yolo_stats: Dict[str, Any] = {}

    if output_format in {"canonical", "both", "yolo"}:
        logger.info("-" * 60)
        logger.info("WRITING CANONICAL DATASET")
        logger.info("-" * 60)
        canonical_stats = _materialize_canonical_dataset(
            canonical_root=canonical_root,
            samples=all_samples,
            schema=schema,
            source_stats=_sanitize_source_stats(source_stats),
            sources_cfg=enabled,
            curation_summary=curation_stats,
        )

    if output_format in {"yolo", "both"}:
        logger.info("-" * 60)
        logger.info("EXPORTING YOLO DATASET")
        logger.info("-" * 60)
        yolo_stats = export_yolo_from_canonical(canonical_root, yolo_root)
        logger.info(
            "  Exported YOLO dataset: %s train + %s val images",
            yolo_stats.get("train_images", 0),
            yolo_stats.get("val_images", 0),
        )

    elapsed = time.time() - start_time
    report_path = generate_report(
        canonical_root,
        source_stats,
        canonical_stats,
        yolo_stats,
        curation_stats,
        elapsed,
    )

    logger.info("=" * 60)
    logger.info("MATERIALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("  Canonical dataset: %s", canonical_root)
    if yolo_stats:
        logger.info("  YOLO dataset:      %s", yolo_root)
    logger.info("  Images:            %s", canonical_stats.get("total_images", 0))
    logger.info("  Elapsed:           %.1fs", elapsed)
    logger.info("  Report:            %s", report_path)

    if curation_stats:
        logger.info("  Curation dropped:  %s images / %s objects", curation_stats.get("dropped_images", 0), curation_stats.get("dropped_objects", 0))

    logger.info("")
    logger.info("  CLASS DISTRIBUTION:")
    class_dist = canonical_stats.get("class_distribution", {})
    for cls_id in sorted(class_dist.keys()):
        cls_name = schema["names"].get(cls_id, f"class_{cls_id}")
        count = class_dist[cls_id]
        bar = "#" * min(50, max(1, count // 100))
        logger.info("    %2d | %-20s | %7d | %s", cls_id, cls_name, count, bar)

    missing_classes = set(range(schema["nc"])) - set(class_dist.keys())
    if missing_classes:
        logger.warning("")
        logger.warning(
            "  Missing classes (no data): %s",
            [schema["names"][cls_id] for cls_id in sorted(missing_classes)],
        )

    logger.info("")
    logger.info("  Next steps:")
    logger.info("    1. Validate canonical: python scripts/validate.py --dataset hf_dataset --format canonical")
    logger.info("    2. Validate YOLO:      python scripts/validate.py --dataset krishi_bouncer_dataset --format yolo")
    logger.info("    3. Package/upload:     python scripts/package_hf_dataset.py --input hf_dataset --repo-id <repo>")


def main():
    parser = argparse.ArgumentParser(
        description="Krishi Vaidya canonical dataset materialization pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "sources.yaml",
        help="Path to sources.yaml configuration file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if a valid canonical dataset already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print plan without downloading or writing",
    )
    parser.add_argument(
        "--output-format",
        choices=("canonical", "yolo", "both"),
        default="both",
        help="Which materialized outputs to produce",
    )

    args = parser.parse_args()
    materialize(
        args.config,
        force=args.force,
        dry_run=args.dry_run,
        output_format=args.output_format,
    )


if __name__ == "__main__":
    main()
