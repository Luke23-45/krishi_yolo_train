"""
scripts/materialize_bouncer.py
-------------------------------
Krishi Vaidya — Bouncer Dataset Materialization Pipeline.

Modeled after the SPECTRA Tiered Acquisition pattern:
    Tier 0: Local integrity check (skip if valid dataset exists)
    Tier 1: Auto-fetch from Kaggle via kagglehub
    Tier 2: Ingest from local paths (user's manual Roboflow downloads)

Usage:
    python scripts/materialize_bouncer.py --config configs/sources.yaml

    # Force re-download and rebuild:
    python scripts/materialize_bouncer.py --config configs/sources.yaml --force

    # Dry-run (validate config without downloading):
    python scripts/materialize_bouncer.py --config configs/sources.yaml --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Resolve project root so imports work when invoked from any directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.adapters import get_adapter  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("krishi.materialize")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================================
# 0. CONFIGURATION LOADER
# ============================================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the sources.yaml configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate required keys
    required = ("schema", "output", "sources")
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: '{key}'")

    schema = cfg["schema"]
    if "nc" not in schema or "names" not in schema:
        raise ValueError("Config 'schema' must contain 'nc' and 'names'")

    if len(schema["names"]) != schema["nc"]:
        raise ValueError(
            f"Schema declares nc={schema['nc']} but names has "
            f"{len(schema['names'])} entries"
        )

    return cfg


# ============================================================================
# 1. TIER 0 — LOCAL INTEGRITY CHECK
# ============================================================================

def check_local_integrity(output_dir: Path, schema: Dict) -> bool:
    """
    Verify that a valid, complete dataset already exists.
    Returns True if the dataset is ready to use.
    """
    data_yaml = output_dir / "data.yaml"
    if not data_yaml.exists():
        return False

    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Existing data.yaml could not be read, rebuilding: {e}")
        return False

    if existing.get("nc") != schema["nc"]:
        logger.warning("Existing dataset has different class count, rebuilding")
        return False

    existing_names = existing.get("names", {})
    if existing_names != schema["names"]:
        logger.warning("Existing dataset class names do not match schema, rebuilding")
        return False

    total_aligned = 0

    for split in ("train", "val"):
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            logger.warning(f"Existing dataset missing split directories for '{split}', rebuilding")
            return False

        images = {
            f.stem: f
            for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        }
        labels = {
            f.stem: f
            for f in lbl_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        }

        if not images or not labels:
            logger.warning(f"Existing dataset split '{split}' is empty, rebuilding")
            return False

        if images.keys() != labels.keys():
            logger.warning(f"Existing dataset split '{split}' has image/label misalignment, rebuilding")
            return False

        for stem, label_path in labels.items():
            try:
                content = label_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Could not read label file {label_path.name}, rebuilding: {e}")
                return False

            if not content:
                logger.warning(f"Empty label file detected ({label_path.name}), rebuilding")
                return False

            for line_no, line in enumerate(content.splitlines(), 1):
                parts = line.split()
                if len(parts) != 5:
                    logger.warning(
                        f"Malformed label line in {label_path.name}:{line_no}, rebuilding"
                    )
                    return False
                try:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                except ValueError:
                    logger.warning(
                        f"Non-numeric label content in {label_path.name}:{line_no}, rebuilding"
                    )
                    return False

                if cls_id < 0 or cls_id >= schema["nc"]:
                    logger.warning(
                        f"Out-of-range class ID in {label_path.name}:{line_no}, rebuilding"
                    )
                    return False

                if any(coord < 0.0 or coord > 1.0 for coord in coords):
                    logger.warning(
                        f"Out-of-range YOLO coordinate in {label_path.name}:{line_no}, rebuilding"
                    )
                    return False

            total_aligned += 1

    if total_aligned == 0:
        logger.warning("Existing dataset contains no aligned samples, rebuilding")
        return False

    return True


# ============================================================================
# 2. TIER 1 — KAGGLE AUTO-FETCH
# ============================================================================

def fetch_kaggle(handle: str, cache_dir: Path) -> Path:
    """
    Download a Kaggle dataset via kagglehub.
    Returns the local path to the downloaded dataset.
    """
    try:
        import kagglehub
    except ImportError:
        raise RuntimeError(
            "kagglehub is required for Kaggle downloads. "
            "Install with: pip install kagglehub"
        )

    logger.info(f"  ↓ Downloading from Kaggle: {handle}")
    start = time.time()

    try:
        local_path = kagglehub.dataset_download(handle)
        elapsed = time.time() - start
        logger.info(f"  ✓ Downloaded in {elapsed:.1f}s → {local_path}")
        return Path(local_path)
    except Exception as e:
        raise RuntimeError(f"Kaggle download failed for '{handle}': {e}")


# ============================================================================
# 3. TIER 2 — ROBOFLOW AUTO-FETCH
# ============================================================================

def fetch_roboflow(handle: str, cache_dir: Path) -> Path:
    """
    Download a Roboflow dataset via the python SDK.
    `handle` format expected: "workspace/project/version"
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY environment variable is required for Roboflow downloads. "
            "Get it from your Roboflow account settings."
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise RuntimeError(
            "roboflow is required. Install with: pip install roboflow"
        )
    
    parts = handle.split("/")
    if len(parts) != 3:
        raise ValueError(f"Roboflow handle must be 'workspace/project/version', got: '{handle}'")
    
    workspace, project_name, version_num = parts
    
    # We download into a dedicated roboflow cache directory
    output_dir = cache_dir / "roboflow"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"  ↓ Fetching Roboflow dataset: {handle}")
    start = time.time()
    
    # Save current working directory
    original_cwd = os.getcwd()
    
    try:
        # Change to cache dir so the Roboflow SDK safely places the dataset here
        os.chdir(output_dir)
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project_name)
        
        # Download natively
        dataset = project.version(int(version_num)).download("yolov8")
        elapsed = time.time() - start
        
        # dataset.location returns the absolute path where it was saved
        logger.info(f"  ✓ Fetched in {elapsed:.1f}s → {dataset.location}")
        final_path = Path(dataset.location)
    except Exception as e:
        raise RuntimeError(f"Roboflow download failed for '{handle}': {e}")
    finally:
        # ALWAYS restore original working directory
        os.chdir(original_cwd)

    return final_path


# ============================================================================
# 4. TIER 3 — LOCAL INGEST MUST EXIST
# ============================================================================

def validate_local_path(path_str: str, source_name: str) -> Path:
    """Validate that a user-provided local path exists and has content."""
    path = Path(path_str)

    if path_str == "REPLACE_WITH_LOCAL_PATH":
        raise ValueError(
            f"Source '{source_name}' has a placeholder path. "
            f"Either provide the real path in sources.yaml or set enabled: false"
        )

    if not path.exists():
        raise FileNotFoundError(
            f"Source '{source_name}' path does not exist: {path}"
        )

    if not path.is_dir():
        raise ValueError(
            f"Source '{source_name}' path is not a directory: {path}"
        )

    return path


# ============================================================================
# 5. MERGE — COMBINE STAGING INTO FINAL DATASET
# ============================================================================

def merge_staging_to_final(
    staging_dir: Path,
    output_dir: Path,
    schema: Dict,
) -> Dict[str, int]:
    """
    Move all staged files into the final dataset directory
    and generate data.yaml.
    """
    logger.info("─" * 60)
    logger.info("MERGE PHASE")
    logger.info("─" * 60)

    # Create output structure
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    merge_stats = {"train_images": 0, "val_images": 0, "total_labels": 0}

    for split in ("train", "val"):
        # Images
        src_img_dir = staging_dir / "images" / split
        dst_img_dir = output_dir / "images" / split

        if src_img_dir.exists():
            for img_file in src_img_dir.iterdir():
                if img_file.is_file():
                    dst = dst_img_dir / img_file.name
                    # Handle duplicates by adding suffix
                    if dst.exists():
                        stem = img_file.stem
                        suffix = img_file.suffix
                        counter = 1
                        while dst.exists():
                            dst = dst_img_dir / f"{stem}__dup{counter}{suffix}"
                            counter += 1
                    shutil.move(str(img_file), str(dst))
                    merge_stats[f"{split}_images"] += 1

        # Labels
        src_lbl_dir = staging_dir / "labels" / split
        dst_lbl_dir = output_dir / "labels" / split

        if src_lbl_dir.exists():
            for lbl_file in src_lbl_dir.iterdir():
                if lbl_file.is_file():
                    dst = dst_lbl_dir / lbl_file.name
                    if dst.exists():
                        stem = lbl_file.stem
                        suffix = lbl_file.suffix
                        counter = 1
                        while dst.exists():
                            dst = dst_lbl_dir / f"{stem}__dup{counter}{suffix}"
                            counter += 1
                    shutil.move(str(lbl_file), str(dst))
                    merge_stats["total_labels"] += 1

    # Generate data.yaml
    data_yaml_content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": schema["nc"],
        "names": schema["names"],
    }

    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml_content, f, default_flow_style=False, allow_unicode=True)

    logger.info(
        f"  Merged: {merge_stats['train_images']} train + "
        f"{merge_stats['val_images']} val images, "
        f"{merge_stats['total_labels']} labels"
    )
    logger.info(f"  data.yaml → {data_yaml_path}")

    return merge_stats


# ============================================================================
# 6. REPORT — GENERATE MATERIALIZATION AUDIT TRAIL
# ============================================================================

def generate_report(
    output_dir: Path,
    source_stats: List[Dict],
    merge_stats: Dict,
    elapsed: float,
) -> Path:
    """Write a JSON report of the entire materialization run."""
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(output_dir.resolve()),
        "elapsed_seconds": round(elapsed, 1),
        "sources": source_stats,
        "merge": merge_stats,
    }

    # Compute totals
    total_images = sum(s.get("images_processed", 0) for s in source_stats)
    total_errors = sum(s.get("errors", 0) for s in source_stats)
    all_unmapped = set()
    for s in source_stats:
        all_unmapped.update(s.get("unmapped_classes", []))

    report["totals"] = {
        "images_processed": total_images,
        "errors": total_errors,
        "unmapped_classes": sorted(all_unmapped),
    }

    # Aggregate class distribution
    class_dist: Dict[int, int] = {}
    for s in source_stats:
        for cls_id, count in s.get("classes_found", {}).items():
            cls_id_int = int(cls_id)
            class_dist[cls_id_int] = class_dist.get(cls_id_int, 0) + count
    report["class_distribution"] = dict(sorted(class_dist.items()))

    report_path = output_dir / "materialization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"  Report → {report_path}")
    return report_path


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def materialize(config_path: Path, force: bool = False, dry_run: bool = False) -> None:
    """
    Main entry point. Orchestrates the full materialization pipeline.
    """
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║   KRISHI VAIDYA — Bouncer Dataset Materialization v1.0  ║
    ║   Tiered Acquisition Pipeline                           ║
    ╚══════════════════════════════════════════════════════════╝
    """
    logger.info(banner)

    start_time = time.time()

    # Load config
    cfg = load_config(config_path)
    schema = cfg["schema"]
    output_cfg = cfg["output"]
    sources = cfg["sources"]

    output_dir = PROJECT_ROOT / output_cfg["root"]
    staging_dir = PROJECT_ROOT / ".staging"
    val_ratio = output_cfg.get("val_split_ratio", 0.2)

    # Count enabled sources
    enabled = [s for s in sources if s.get("enabled", True)]
    logger.info(f"Config loaded: {len(enabled)}/{len(sources)} sources enabled")
    logger.info(f"Schema: {schema['nc']} classes")
    logger.info(f"Output: {output_dir}")

    if dry_run:
        logger.info("─" * 60)
        logger.info("DRY RUN — validating config only, no downloads or writes")
        logger.info("─" * 60)
        for src in enabled:
            logger.info(f"  [{src['type'].upper():6s}] {src['name']} — {src.get('handle', 'N/A')}")
        logger.info("Config is valid. Exiting dry-run.")
        return

    # ── TIER 0: Local integrity check ──
    if not force and check_local_integrity(output_dir, schema):
        logger.info("─" * 60)
        logger.info("TIER 0 │ Valid dataset found locally. Skipping materialization.")
        logger.info(f"        │ To force rebuild, run with --force")
        logger.info("─" * 60)
        return

    # ── Clean staging area ──
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    for split in ("train", "val"):
        (staging_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (staging_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Process each enabled source ──
    source_stats = []

    for i, src_cfg in enumerate(enabled):
        name = src_cfg["name"]
        src_type = src_cfg["type"]
        handle = src_cfg.get("handle", "")
        fmt = src_cfg["format"]
        split_strategy = src_cfg.get("split", "preserve")
        class_map = src_cfg.get("class_map", {})

        logger.info("─" * 60)
        logger.info(f"SOURCE {i + 1}/{len(enabled)} │ {name}")
        logger.info(f"  Type: {src_type} │ Format: {fmt} │ Split: {split_strategy}")
        logger.info("─" * 60)

        try:
            # Resolve source directory
            if src_type == "kaggle":
                source_dir = fetch_kaggle(handle, PROJECT_ROOT / ".cache")
            elif src_type == "roboflow":
                source_dir = fetch_roboflow(handle, PROJECT_ROOT / ".cache")
            elif src_type == "local":
                source_dir = validate_local_path(handle, name)
            else:
                raise ValueError(f"Unknown source type: '{src_type}'")

            # Get the appropriate adapter
            adapter_cls = get_adapter(fmt)
            adapter = adapter_cls()

            # Process
            stats = adapter.process(
                source_dir=source_dir,
                staging_dir=staging_dir,
                class_map=class_map,
                source_name=name,
                split_strategy=split_strategy,
                val_ratio=val_ratio,
            )

            stats["source_name"] = name
            stats["source_type"] = src_type
            stats["source_handle"] = handle
            source_stats.append(stats)

        except Exception as e:
            logger.error(f"  ✗ FAILED: {e}")
            source_stats.append({
                "source_name": name,
                "source_type": src_type,
                "source_handle": handle,
                "images_processed": 0,
                "errors": 1,
                "error_message": str(e),
            })
            continue

    # ── Check if we got any data ──
    total_processed = sum(s.get("images_processed", 0) for s in source_stats)
    if total_processed == 0:
        logger.error("═" * 60)
        logger.error("FATAL: No images were processed from any source.")
        logger.error("Check your sources.yaml — are paths correct? Are sources enabled?")
        logger.error("═" * 60)
        # Clean up staging
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        sys.exit(1)

    # ── MERGE ──
    # Clean previous output if force-rebuilding
    if output_dir.exists():
        logger.info("Cleaning previous output directory before final merge")
        shutil.rmtree(output_dir)

    merge_stats = merge_staging_to_final(staging_dir, output_dir, schema)

    # ── REPORT ──
    elapsed = time.time() - start_time
    report_path = generate_report(output_dir, source_stats, merge_stats, elapsed)

    # ── Clean staging ──
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    # ── Summary ──
    logger.info("═" * 60)
    logger.info("MATERIALIZATION COMPLETE")
    logger.info("═" * 60)
    logger.info(f"  Dataset:  {output_dir}")
    logger.info(f"  Images:   {total_processed}")
    logger.info(f"  Train:    {merge_stats.get('train_images', 0)}")
    logger.info(f"  Val:      {merge_stats.get('val_images', 0)}")
    logger.info(f"  Elapsed:  {elapsed:.1f}s")
    logger.info(f"  Report:   {report_path}")

    # Print class distribution
    logger.info("")
    logger.info("  CLASS DISTRIBUTION:")
    all_classes: Dict[int, int] = {}
    for s in source_stats:
        for cls_id, count in s.get("classes_found", {}).items():
            cls_id_int = int(cls_id)
            all_classes[cls_id_int] = all_classes.get(cls_id_int, 0) + count

    for cls_id in sorted(all_classes.keys()):
        cls_name = schema["names"].get(cls_id, f"class_{cls_id}")
        count = all_classes[cls_id]
        bar = "█" * min(50, count // 100)
        logger.info(f"    {cls_id:2d} │ {cls_name:20s} │ {count:>7,d} │ {bar}")

    missing_classes = set(range(schema["nc"])) - set(all_classes.keys())
    if missing_classes:
        logger.warning("")
        logger.warning(
            f"  ⚠ Missing classes (no data): "
            f"{[schema['names'][c] for c in sorted(missing_classes)]}"
        )
        logger.warning(
            "  Enable additional sources in sources.yaml or provide Roboflow downloads."
        )

    logger.info("")
    logger.info("  Next steps:")
    logger.info("    1. Run: python scripts/validate.py --dataset krishi_bouncer_dataset")
    logger.info("    2. Enable more sources in configs/sources.yaml")
    logger.info("    3. Train: yolo detect train data=krishi_bouncer_dataset/data.yaml model=yolov8n.pt")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Krishi Vaidya — Bouncer Dataset Materialization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/materialize_bouncer.py --config configs/sources.yaml
    python scripts/materialize_bouncer.py --config configs/sources.yaml --force
    python scripts/materialize_bouncer.py --config configs/sources.yaml --dry-run
        """,
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
        help="Force rebuild even if a valid dataset already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print plan without downloading or writing",
    )

    args = parser.parse_args()
    materialize(args.config, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
