#!/usr/bin/env python
"""
scripts/run_all_validations.py
------------------------------
End-to-End Orchestrator for YOLO Quantization and Validation with State Machine.

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
from datetime import datetime
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
    
    # New state machine / selection arguments
    parser.add_argument(
        "--run-steps", 
        default=None, 
        help="Comma-separated list of validation steps to run. E.g., 'extract,quantize'. If omitted, all steps are run."
    )
    parser.add_argument(
        "--skip-steps", 
        default=None, 
        help="Comma-separated list of validation steps to skip. E.g., 'baseline_latency,baseline_robust'."
    )
    parser.add_argument(
        "--force-quantize", 
        action="store_true", 
        help="Force quantization run even if quantized outputs directory already exists."
    )
    parser.add_argument(
        "--skip-baseline", 
        action="store_true", 
        help="Skip extraction and baseline validations to resume directly from quantization (legacy alias)"
    )
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


class ValidationContext:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_path = Path(args.model).resolve()
        self.data_path = Path(args.data).resolve()
        self.run_dir = Path(args.run_dir).resolve()
        self.images_dir = Path(args.images_dir).resolve()
        self.output_root = Path(args.output_root).resolve()
        self.imgsz = args.imgsz
        self.device = args.device
        self.conf = args.conf
        self.iou = args.iou
        
        # Resolve active quantized directory
        self.quantize_out = self._resolve_quantize_dir(args.force_quantize)

    def _resolve_quantize_dir(self, force_quantize: bool) -> Path:
        """Dynamically resolve the quantized output directory path.
        
        If force_quantize is requested, creates a new unique directory with a timestamp.
        Otherwise, scans output_root to locate the most recently modified valid quantized directory
        (defaulting to output_root / 'quantized' if none are found).
        """
        default_dir = self.output_root / "quantized"
        
        if force_quantize:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            forced_dir = self.output_root / f"quantized_{timestamp}"
            logger.info(f"Force-quantize requested. Resolved new quantized output directory: {forced_dir}")
            return forced_dir
            
        candidates = []
        if self.output_root.exists():
            for p in self.output_root.iterdir():
                if p.is_dir() and (p.name == "quantized" or p.name.startswith("quantized_")):
                    # Validate if it actually has quantized models or a manifest
                    has_manifest = (p / "quant_manifest.json").exists()
                    has_tflite = any((p / level).exists() for level in ["fp32", "fp16", "int8"])
                    if has_manifest or has_tflite:
                        candidates.append((p.stat().st_mtime, p))
                        
        if candidates:
            # Sort candidates by modification time, most recent first
            candidates.sort(key=lambda x: x[0], reverse=True)
            newest_dir = candidates[0][1]
            logger.info(f"Found existing valid quantized directory: {newest_dir}")
            return newest_dir
            
        logger.info(f"No existing quantized directory found. Defaulting to: {default_dir}")
        return default_dir


class ValidationStateMachine:
    def __init__(self, ctx: ValidationContext):
        self.ctx = ctx
        self.steps = {
            "extract": self.run_extract,
            "baseline_threshold": self.run_baseline_threshold,
            "baseline_latency": self.run_baseline_latency,
            "baseline_robust": self.run_baseline_robust,
            "quantize": self.run_quantize,
            "quantized_latency": self.run_quantized_latency,
            "quantized_robust": self.run_quantized_robust,
        }
        
    def get_active_steps(self) -> List[str]:
        """Resolves which steps should be run based on run-steps and skip-steps inputs."""
        # 1. Parse run-steps
        if self.ctx.args.run_steps:
            run_list = [s.strip() for s in self.ctx.args.run_steps.split(",") if s.strip()]
            invalid = [s for s in run_list if s not in self.steps]
            if invalid:
                raise ValueError(f"Invalid step(s) specified in --run-steps: {invalid}. Available: {list(self.steps.keys())}")
            active = run_list
        else:
            active = list(self.steps.keys())
            
        # 2. Parse skip-steps
        skip_list = []
        if self.ctx.args.skip_steps:
            skip_list = [s.strip() for s in self.ctx.args.skip_steps.split(",") if s.strip()]
            invalid = [s for s in skip_list if s not in self.steps]
            if invalid:
                raise ValueError(f"Invalid step(s) specified in --skip-steps: {invalid}. Available: {list(self.steps.keys())}")
                
        # 3. Handle legacy --skip-baseline alias
        if self.ctx.args.skip_baseline:
            baseline_steps = ["extract", "baseline_threshold", "baseline_latency", "baseline_robust"]
            skip_list.extend(baseline_steps)
            
        # 4. Filter active steps
        active = [s for s in active if s not in skip_list]
        return active

    def execute(self) -> int:
        """Orchestrates and executes the active steps in the validation pipeline."""
        try:
            active_steps = self.get_active_steps()
        except ValueError as e:
            logger.error(str(e))
            return 1
            
        if not active_steps:
            logger.warning("No steps selected for execution.")
            return 0
            
        logger.info(f"Active validation pipeline steps: {active_steps}")
        overall_success = True
        
        for step_name in active_steps:
            logger.info(f"==================================================")
            logger.info(f"Executing Step: {step_name}")
            logger.info(f"==================================================")
            
            step_func = self.steps[step_name]
            try:
                step_success = step_func()
                if not step_success:
                    logger.error(f"Step '{step_name}' failed.")
                    overall_success = False
            except Exception as e:
                logger.exception(f"Step '{step_name}' raised an unhandled exception: {e}")
                overall_success = False
                
        if overall_success:
            logger.info("==================================================")
            logger.info("Validation pipeline completed successfully!")
            logger.info(f"Results saved to: {self.ctx.output_root}")
            logger.info("==================================================")
            return 0
        else:
            logger.warning("==================================================")
            logger.warning("Validation pipeline finished with errors. Check logs.")
            logger.warning("==================================================")
            return 1

    def run_extract(self) -> bool:
        thesis_out = self.ctx.output_root / "thesis_data"
        cmd = [
            sys.executable, "-m", "yoloml.validations.extract_thesis_data",
            f"run_dir={self.ctx.run_dir}",
            f"output_dir={thesis_out}"
        ]
        return run_command(cmd, "Extract Thesis Data")
        
    def run_baseline_threshold(self) -> bool:
        baseline_out = self.ctx.output_root / "baseline"
        cmd = [
            sys.executable, "-m", "yoloml.validations.threshold_analysis",
            f"model={self.ctx.model_path}",
            f"data={self.ctx.data_path}",
            f"output_dir={baseline_out / 'threshold'}",
            f"imgsz={self.ctx.imgsz}",
            f"device={self.ctx.device}"
        ]
        return run_command(cmd, "Baseline Threshold Analysis")
        
    def run_baseline_latency(self) -> bool:
        baseline_out = self.ctx.output_root / "baseline"
        cmd = [
            sys.executable, "-m", "yoloml.validations.benchmark_latency",
            f"model={self.ctx.model_path}",
            f"images_dir={self.ctx.images_dir}",
            f"output_dir={baseline_out / 'latency'}",
            f"imgsz={self.ctx.imgsz}",
            f"device={self.ctx.device}"
        ]
        return run_command(cmd, "Baseline Latency Benchmark")
        
    def run_baseline_robust(self) -> bool:
        baseline_out = self.ctx.output_root / "baseline"
        cmd = [
            sys.executable, "-m", "yoloml.validations.robust_validate",
            "--model", str(self.ctx.model_path),
            "--data", str(self.ctx.data_path),
            "--images-dir", str(self.ctx.images_dir),
            "--output-dir", str(baseline_out / 'robust_validation'),
            "--imgsz", str(self.ctx.imgsz),
            "--conf", str(self.ctx.conf),
            "--iou", str(self.ctx.iou),
            "--device", self.ctx.device
        ]
        return run_command(cmd, "Baseline Robust Validation")

    def run_quantize(self) -> bool:
        # Check if the resolved directory already exists and has models
        # (This check will be skipped naturally if force-quantize creates a new non-existing timestamped directory)
        if self.ctx.quantize_out.exists() and not self.ctx.args.force_quantize:
            has_models = any((self.ctx.quantize_out / lvl).exists() for lvl in ["fp32", "fp16", "int8"])
            if has_models:
                logger.info(f"Quantization cache hit! Valid models found in: {self.ctx.quantize_out}")
                logger.info("Skipping quantization step execution.")
                return True
                
        # Perform quantization
        self.ctx.quantize_out.mkdir(parents=True, exist_ok=True)
        levels = ["fp32", "fp16", "int8"]
        success = True
        for level in levels:
            cmd = [
                sys.executable, "-m", "yoloml.models.quantize",
                f"++quantization.model={self.ctx.model_path}",
                f"++quantization.data={self.ctx.data_path}",
                f"++quantization.output_root={self.ctx.quantize_out}",
                f"++quantization.imgsz={self.ctx.imgsz}",
                f"++quantization.levels=[{level}]"
            ]
            if not run_command(cmd, f"Quantization ({level.upper()})"):
                logger.error(f"Quantization failed for {level}.")
                success = False
        return success

    def _get_quantized_models(self) -> List[Dict[str, str]]:
        # Helper to load all quantized models found inside resolved quantize_out
        levels = ["fp32", "fp16", "int8"]
        models = []
        for level in levels:
            level_dir = self.ctx.quantize_out / level
            metadata_file = level_dir / "metadata.json"
            
            if not metadata_file.exists():
                logger.warning(f"Metadata not found for {level} in {level_dir}. Skipping.")
                continue
                
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                tflite_path = metadata.get("tflite_path")
                if tflite_path and Path(tflite_path).exists():
                    models.append({
                        "level": level,
                        "path": tflite_path,
                        "dir": level_dir
                    })
                else:
                    logger.warning(f"TFLite model path '{tflite_path}' missing or invalid for {level}.")
            except Exception as e:
                logger.error(f"Failed to read metadata for {level}: {e}")
        return models

    def run_quantized_latency(self) -> bool:
        models = self._get_quantized_models()
        if not models:
            logger.error("No valid quantized models found for validation. Ensure 'quantize' step is run first.")
            return False
            
        success = True
        for model_info in models:
            level = model_info["level"]
            tflite_path = model_info["path"]
            level_dir = model_info["dir"]
            
            cmd = [
                sys.executable, "-m", "yoloml.validations.benchmark_latency",
                f"model={tflite_path}",
                f"images_dir={self.ctx.images_dir}",
                f"output_dir={level_dir / 'latency'}",
                f"imgsz={self.ctx.imgsz}",
                f"device={self.ctx.device}"
            ]
            if not run_command(cmd, f"Quantized ({level}) Latency Benchmark"):
                success = False
        return success

    def run_quantized_robust(self) -> bool:
        models = self._get_quantized_models()
        if not models:
            logger.error("No valid quantized models found for validation. Ensure 'quantize' step is run first.")
            return False
            
        success = True
        for model_info in models:
            level = model_info["level"]
            tflite_path = model_info["path"]
            level_dir = model_info["dir"]
            
            cmd = [
                sys.executable, "-m", "yoloml.validations.robust_validate",
                "--model", str(tflite_path),
                "--data", str(self.ctx.data_path),
                "--images-dir", str(self.ctx.images_dir),
                "--output-dir", str(level_dir / 'robust_validation'),
                "--imgsz", str(self.ctx.imgsz),
                "--conf", str(self.ctx.conf),
                "--iou", str(self.ctx.iou),
                "--device", self.ctx.device
            ]
            if not run_command(cmd, f"Quantized ({level}) Robust Validation"):
                success = False
        return success


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

    ctx = ValidationContext(args)
    state_machine = ValidationStateMachine(ctx)
    return state_machine.execute()


if __name__ == "__main__":
    sys.exit(main())
