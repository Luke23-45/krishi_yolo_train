#!/usr/bin/env python
"""
scripts/run_all_validations.py
------------------------------
End-to-End Orchestrator for YOLO Quantization and Validation.

This script automates:
1. Extracting thesis-ready data from a training run.
2. Validating the baseline PyTorch model (threshold analysis, latency, accuracy).
3. Quantizing the model to FP32, FP16, and INT8 formats.
4. Validating each quantized model (latency, accuracy).
5. Persisting all data in a structured, hierarchical directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_all_validations")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-End YOLO Quantization & Validation Orchestrator")
    parser.add_argument("--model", required=True, help="Path to baseline model (e.g., best.pt)")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--run-dir", required=True, help="Path to original training run directory (for thesis data)")
    parser.add_argument("--images-dir", required=True, help="Path to validation images directory")
    parser.add_argument("--output-root", required=True, help="Master directory for all persistent results")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="cpu", help="Inference device (e.g., cpu, cuda:0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for robust validation")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for robust validation")
    return parser


def run_command(cmd: List[str], step_name: str) -> bool:
    """Run a subprocess command and stream output."""
    logger.info(f"--- Starting {step_name} ---")
    logger.info(f"Command: {' '.join(cmd)}")
    start_time = time.perf_counter()
    
    import os
    env = os.environ.copy()
    src_path = str(PROJECT_ROOT / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_path
    
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, env=env)
        elapsed = time.perf_counter() - start_time
        logger.info(f"--- Completed {step_name} in {elapsed:.2f}s ---\n")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start_time
        logger.error(f"--- FAILED {step_name} in {elapsed:.2f}s (Exit code: {e.returncode}) ---\n")
        return False


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    data_path = Path(args.data).resolve()
    run_dir = Path(args.run_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    output_root = Path(args.output_root).resolve()

    if not model_path.exists():
        logger.error(f"Baseline model not found: {model_path}")
        return 1
    if not data_path.exists():
        logger.error(f"Data YAML not found: {data_path}")
        return 1
    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return 1
    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Master output directory: {output_root}")

    # Modules for scripts
    extract_mod = "yoloml.validations.extract_thesis_data"
    threshold_mod = "yoloml.validations.threshold_analysis"
    benchmark_mod = "yoloml.validations.benchmark_latency"
    robust_mod = "yoloml.validations.robust_validate"
    quantize_mod = "yoloml.models.quantize"

    overall_success = True

    # 1. Extract Thesis Data
    thesis_out = output_root / "thesis_data"
    cmd_extract = [
        sys.executable, "-m", extract_mod,
        f"run_dir={run_dir}",
        f"output_dir={thesis_out}"
    ]
    if not run_command(cmd_extract, "Extract Thesis Data"):
        overall_success = False

    # 2. Baseline Validations
    baseline_out = output_root / "baseline"
    
    # 2a. Threshold Analysis
    cmd_thresh = [
        sys.executable, "-m", threshold_mod,
        f"model={model_path}",
        f"data={data_path}",
        f"output_dir={baseline_out / 'threshold'}",
        f"imgsz={args.imgsz}",
        f"device={args.device}"
    ]
    if not run_command(cmd_thresh, "Baseline Threshold Analysis"):
        overall_success = False

    # 2b. Benchmark Latency
    cmd_bench = [
        sys.executable, "-m", benchmark_mod,
        f"model={model_path}",
        f"images_dir={images_dir}",
        f"output_dir={baseline_out / 'latency'}",
        f"imgsz={args.imgsz}",
        f"device={args.device}"
    ]
    if not run_command(cmd_bench, "Baseline Latency Benchmark"):
        overall_success = False

    # 2c. Robust Validate
    cmd_robust = [
        sys.executable, "-m", robust_mod,
        "--model", str(model_path),
        "--data", str(data_path),
        "--images-dir", str(images_dir),
        "--output-dir", str(baseline_out / 'robust_validation'),
        "--imgsz", str(args.imgsz),
        "--conf", str(args.conf),
        "--iou", str(args.iou),
        "--device", args.device
    ]
    if not run_command(cmd_robust, "Baseline Robust Validation"):
        overall_success = False

    # 3. Quantization
    quantize_out = output_root / "quantized"
    cmd_quantize = [
        sys.executable, "-m", quantize_mod,
        f"quantization.model={model_path}",
        f"quantization.data={data_path}",
        f"quantization.output_root={quantize_out}",
        f"quantization.imgsz={args.imgsz}",
        f"quantization.levels=[fp32,fp16,int8]"
    ]
    if not run_command(cmd_quantize, "Multi-level Quantization"):
        logger.error("Quantization failed. Skipping quantized validations.")
        return 1

    # 4. Quantized Validations
    # quantize.py outputs to `quantize_out / {level}` and writes a metadata.json
    levels = ["fp32", "fp16", "int8"]
    for level in levels:
        level_dir = quantize_out / level
        metadata_file = level_dir / "metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"Metadata not found for {level}. Skipping validation for {level}.")
            continue
            
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            tflite_path = metadata.get("tflite_path")
            if not tflite_path or not Path(tflite_path).exists():
                logger.warning(f"TFLite model missing for {level}. Skipping validation.")
                continue
        except Exception as e:
            logger.error(f"Failed to read metadata for {level}: {e}")
            continue
            
        logger.info(f"Validating Quantized Model: {level} ({tflite_path})")

        # 4a. Quantized Latency
        cmd_qbench = [
            sys.executable, "-m", benchmark_mod,
            f"model={tflite_path}",
            f"images_dir={images_dir}",
            f"output_dir={level_dir / 'latency'}",
            f"imgsz={args.imgsz}",
            f"device={args.device}"
        ]
        if not run_command(cmd_qbench, f"Quantized ({level}) Latency Benchmark"):
            overall_success = False

        # 4b. Quantized Robust Validate
        cmd_qrobust = [
            sys.executable, "-m", robust_mod,
            "--model", str(tflite_path),
            "--data", str(data_path),
            "--images-dir", str(images_dir),
            "--output-dir", str(level_dir / 'robust_validation'),
            "--imgsz", str(args.imgsz),
            "--conf", str(args.conf),
            "--iou", str(args.iou),
            "--device", args.device
        ]
        if not run_command(cmd_qrobust, f"Quantized ({level}) Robust Validation"):
            overall_success = False

    if overall_success:
        logger.info(f"All validations completed successfully! Data saved to: {output_root}")
        return 0
    else:
        logger.warning(f"Pipeline completed with errors. Check logs. Partial data saved to: {output_root}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
