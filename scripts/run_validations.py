#!/usr/bin/env python
"""
scripts/run_validations.py
--------------------------
Orchestrator for all YOLO post-training validation scripts.

Run everything:
    python scripts/run_validations.py --all

Run specific steps:
    python scripts/run_validations.py extract
    python scripts/run_validations.py benchmark threshold
    python scripts/run_validations.py robust --device cuda:0

List available steps:
    python scripts/run_validations.py --list

Dry-run (show commands without executing):
    python scripts/run_validations.py --all --dry-run

Override device for all GPU-dependent steps:
    python scripts/run_validations.py --all --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs" / "validation"
OUTPUT_ROOT = PROJECT_ROOT / "krishi_yolo_train_outputs" / "thesis_ready_data"

STEP_ORDER = ["extract", "benchmark", "threshold", "robust"]


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

@dataclass
class ValidationStep:
    """Defines a single validation step."""
    name: str
    description: str
    module: str                       # python -m <module>
    config_file: Optional[str]        # config filename in configs/validation/
    uses_hydra: bool                  # True = Hydra CLI, False = argparse CLI
    requires_val_images: bool         # True = needs extracted val split
    requires_model: bool              # True = needs model weights + ultralytics
    output_dir_name: str              # subdirectory name under thesis_ready_data/
    device_param: Optional[str]       # config key for --device override (None = not applicable)
    extra_args: List[str] = field(default_factory=list)


STEPS: Dict[str, ValidationStep] = {
    "extract": ValidationStep(
        name="extract",
        description="Extract and restructure training metrics into thesis-ready CSVs/JSON",
        module="yoloml.validations.extract_thesis_data",
        config_file="extract_thesis_data.yaml",
        uses_hydra=True,
        requires_val_images=False,
        requires_model=False,
        output_dir_name="extracted",
        device_param=None,
    ),
    "benchmark": ValidationStep(
        name="benchmark",
        description="Benchmark inference latency (per-image timing, FPS, percentiles)",
        module="yoloml.validations.benchmark_latency",
        config_file="benchmark_latency.yaml",
        uses_hydra=True,
        requires_val_images=True,
        requires_model=True,
        output_dir_name="benchmark_latency",
        device_param="device",
    ),
    "threshold": ValidationStep(
        name="threshold",
        description="Sweep confidence thresholds and find optimal operating points",
        module="yoloml.validations.threshold_analysis",
        config_file="threshold_analysis.yaml",
        uses_hydra=True,
        requires_val_images=True,
        requires_model=True,
        output_dir_name="threshold_analysis",
        device_param="device",
    ),
    "robust": ValidationStep(
        name="robust",
        description="Full per-image validation with error analysis and confusion matrix",
        module="yoloml.validations.robust_validate",
        config_file="robust_validation.yaml",
        uses_hydra=False,
        requires_val_images=True,
        requires_model=True,
        output_dir_name="robust_validation",
        device_param="device",
    ),
}


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_model_weights() -> Optional[str]:
    """Check that best.pt exists. Returns error message or None."""
    weights = (
        PROJECT_ROOT
        / "krishi_yolo_train_outputs"
        / "krishi_yolo_train"
        / "outputs"
        / "runs"
        / "20260516_173740"
        / "train"
        / "artifacts"
        / "weights"
        / "best.pt"
    )
    if not weights.exists():
        return f"Model weights not found: {weights}"
    return None


def _check_val_images() -> Optional[str]:
    """Check that the val split has been extracted from webdataset. Returns error or None."""
    val_dir = PROJECT_ROOT / "krishi_bouncer_dataset" / "val"
    if not val_dir.exists():
        return f"Validation directory missing: {val_dir}"

    # Check for images subdirectory or direct images
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    has_images = False
    for child in val_dir.rglob("*"):
        if child.is_file() and child.suffix.lower() in image_suffixes:
            has_images = True
            break
    if not has_images:
        return (
            f"No validation images found in {val_dir}. "
            f"Extract images from the webdataset tar: "
            f"krishi_bouncer_dataset/val/val-000000.tar"
        )
    return None


def _check_data_yaml() -> Optional[str]:
    """Check that data.yaml exists for threshold analysis."""
    data_yaml = PROJECT_ROOT / "krishi_bouncer_dataset" / "data.yaml"
    if not data_yaml.exists():
        return f"data.yaml not found: {data_yaml}"
    return None


def _check_config(step: ValidationStep) -> Optional[str]:
    """Check that the config file for a step exists."""
    if step.config_file is None:
        return None
    config_path = CONFIGS_DIR / step.config_file
    if not config_path.exists():
        return f"Config file missing: {config_path}"
    return None


def run_preflight(steps: List[ValidationStep]) -> List[str]:
    """Run all pre-flight checks for the requested steps. Returns list of errors."""
    errors: List[str] = []

    needs_model = any(s.requires_model for s in steps)
    needs_val = any(s.requires_val_images for s in steps)

    if needs_model:
        err = _check_model_weights()
        if err:
            errors.append(err)

    if needs_val:
        err = _check_val_images()
        if err:
            errors.append(err)

        err = _check_data_yaml()
        if err:
            errors.append(err)

    for step in steps:
        err = _check_config(step)
        if err:
            errors.append(err)

    return errors


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------

def build_command(step: ValidationStep, device_override: Optional[str] = None) -> List[str]:
    """Build the subprocess command for a validation step."""
    cmd = [sys.executable, "-m", step.module]

    if step.uses_hydra:
        # Hydra scripts pick up config via @hydra.main decorator — no extra args needed.
        # Device override via Hydra CLI override syntax.
        if device_override and step.device_param:
            cmd.append(f"{step.device_param}={device_override}")
    else:
        # argparse-based script (robust_validate)
        config_path = CONFIGS_DIR / step.config_file
        cmd.extend(["--config", str(config_path)])
        if device_override and step.device_param:
            cmd.extend(["--device", device_override])

    cmd.extend(step.extra_args)
    return cmd


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of a single validation step."""
    name: str
    status: str              # "success", "failed", "skipped"
    exit_code: Optional[int]
    duration_seconds: float
    output_dir: str
    error_message: Optional[str] = None


def execute_step(
    step: ValidationStep,
    device_override: Optional[str] = None,
    dry_run: bool = False,
) -> StepResult:
    """Execute a single validation step and return the result."""
    output_dir = OUTPUT_ROOT / step.output_dir_name
    cmd = build_command(step, device_override)
    cmd_str = " ".join(cmd)

    print(f"\n{'=' * 72}")
    print(f"  STEP: {step.name}")
    print(f"  {step.description}")
    print(f"  Command: {cmd_str}")
    print(f"  Output:  {output_dir}")
    print(f"{'=' * 72}")

    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return StepResult(
            name=step.name,
            status="skipped",
            exit_code=None,
            duration_seconds=0.0,
            output_dir=str(output_dir),
            error_message="dry run",
        )

    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
        )
        elapsed = time.perf_counter() - start
        status = "success" if result.returncode == 0 else "failed"
        error_msg = f"exit code {result.returncode}" if result.returncode != 0 else None

        return StepResult(
            name=step.name,
            status=status,
            exit_code=result.returncode,
            duration_seconds=round(elapsed, 2),
            output_dir=str(output_dir),
            error_message=error_msg,
        )

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - start
        print(f"\n  [INTERRUPTED] Step '{step.name}' cancelled by user.")
        return StepResult(
            name=step.name,
            status="failed",
            exit_code=-1,
            duration_seconds=round(elapsed, 2),
            output_dir=str(output_dir),
            error_message="interrupted by user",
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return StepResult(
            name=step.name,
            status="failed",
            exit_code=-1,
            duration_seconds=round(elapsed, 2),
            output_dir=str(output_dir),
            error_message=str(exc),
        )


def run_all_steps(
    steps: List[ValidationStep],
    device_override: Optional[str] = None,
    dry_run: bool = False,
    stop_on_failure: bool = False,
) -> List[StepResult]:
    """Execute steps in order and return all results."""
    results: List[StepResult] = []

    total = len(steps)
    for idx, step in enumerate(steps, 1):
        print(f"\n\n{'#' * 72}")
        print(f"  [{idx}/{total}] Running: {step.name}")
        print(f"{'#' * 72}")

        result = execute_step(step, device_override, dry_run)
        results.append(result)

        if result.status == "failed" and stop_on_failure:
            print(f"\n  [ABORT] --stop-on-failure is set. Halting after '{step.name}'.")
            # Mark remaining steps as skipped
            for remaining in steps[idx:]:
                results.append(StepResult(
                    name=remaining.name,
                    status="skipped",
                    exit_code=None,
                    duration_seconds=0.0,
                    output_dir=str(OUTPUT_ROOT / remaining.output_dir_name),
                    error_message="skipped due to prior failure",
                ))
            break

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: List[StepResult], total_elapsed: float) -> None:
    """Print a summary report and save it as JSON."""
    print(f"\n\n{'=' * 72}")
    print("  VALIDATION PIPELINE REPORT")
    print(f"{'=' * 72}")

    status_icons = {"success": "[OK]", "failed": "[FAIL]", "skipped": "[SKIP]"}
    max_name = max(len(r.name) for r in results)

    for r in results:
        icon = status_icons.get(r.status, "[??]")
        duration_str = f"{r.duration_seconds:.1f}s" if r.duration_seconds > 0 else "  -  "
        error_str = f"  ({r.error_message})" if r.error_message else ""
        print(f"  {icon:<6} {r.name:<{max_name}}   {r.status:<8}  {duration_str:>8}{error_str}")

    passed = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    skipped = sum(1 for r in results if r.status == "skipped")
    print(f"\n  Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 72}")

    # Save report JSON
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "steps": [
            {
                "name": r.name,
                "status": r.status,
                "exit_code": r.exit_code,
                "duration_seconds": r.duration_seconds,
                "output_dir": r.output_dir,
                "error": r.error_message,
            }
            for r in results
        ],
    }

    report_path = OUTPUT_ROOT / "validation_run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_cli() -> argparse.ArgumentParser:
    step_names = ", ".join(STEP_ORDER)

    parser = argparse.ArgumentParser(
        description="Run YOLO post-training validation scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
available steps (in execution order):
  {chr(10).join(f'  {s.name:<12} {s.description}' for s in [STEPS[n] for n in STEP_ORDER])}

examples:
  python scripts/run_validations.py --all
  python scripts/run_validations.py extract
  python scripts/run_validations.py benchmark threshold --device cuda:0
  python scripts/run_validations.py --all --dry-run
  python scripts/run_validations.py --list
""",
    )

    parser.add_argument(
        "steps",
        nargs="*",
        default=[],
        metavar="STEP",
        help=f"Steps to run. Choices: {step_names}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all validation steps in order.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_steps",
        help="List all available steps and exit.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override inference device for all steps (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Halt the pipeline if any step fails.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip pre-flight checks (file existence, etc.).",
    )

    return parser


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    # --list: print steps and exit
    if args.list_steps:
        print("\nAvailable validation steps:\n")
        for name in STEP_ORDER:
            s = STEPS[name]
            needs = []
            if s.requires_model:
                needs.append("model")
            if s.requires_val_images:
                needs.append("val images")
            needs_str = f"  (requires: {', '.join(needs)})" if needs else ""
            print(f"  {s.name:<12}  {s.description}{needs_str}")
        print(f"\nConfigs directory : {CONFIGS_DIR}")
        print(f"Output root       : {OUTPUT_ROOT}")
        return

    # Resolve which steps to run
    if args.all:
        selected_names = STEP_ORDER
    elif args.steps:
        # Validate step names
        invalid = [s for s in args.steps if s not in STEPS]
        if invalid:
            parser.error(
                f"Unknown step(s): {', '.join(invalid)}. "
                f"Valid choices: {', '.join(STEP_ORDER)}"
            )
        # Preserve user order but deduplicate
        seen = set()
        selected_names = []
        for name in args.steps:
            if name not in seen:
                seen.add(name)
                selected_names.append(name)
    else:
        parser.error("Specify steps to run (e.g. 'extract benchmark') or use --all.")
        return

    selected_steps = [STEPS[name] for name in selected_names]

    print(f"\n{'#' * 72}")
    print(f"  Krishi Vaidya - Validation Pipeline Runner")
    print(f"  Steps: {', '.join(selected_names)}")
    print(f"  Device override: {args.device or '(use config defaults)'}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"{'#' * 72}")

    # Pre-flight checks
    if not args.skip_preflight and not args.dry_run:
        errors = run_preflight(selected_steps)
        if errors:
            print(f"\n  Pre-flight checks FAILED:\n")
            for err in errors:
                print(f"    ✗  {err}")
            print(f"\n  Use --skip-preflight to bypass or fix the issues above.")
            sys.exit(1)
        print("\n  Pre-flight checks passed. ✓")

    # Execute
    pipeline_start = time.perf_counter()
    results = run_all_steps(
        selected_steps,
        device_override=args.device,
        dry_run=args.dry_run,
        stop_on_failure=args.stop_on_failure,
    )
    pipeline_elapsed = time.perf_counter() - pipeline_start

    # Report
    print_report(results, pipeline_elapsed)

    # Exit code: non-zero if any step failed
    if any(r.status == "failed" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
