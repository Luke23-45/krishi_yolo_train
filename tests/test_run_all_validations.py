import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

# Add project root to python path to import scripts folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_all_validations import ValidationContext, ValidationStateMachine, build_arg_parser

def test_validation_context_directories(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    
    # 1. No existing quantized dir
    parser = build_arg_parser()
    args = parser.parse_args([
        "--model", str(tmp_path / "model.pt"),
        "--data", str(tmp_path / "data.yaml"),
        "--run-dir", str(tmp_path / "run"),
        "--images-dir", str(tmp_path / "images"),
        "--output-root", str(output_root),
    ])
    
    # Create fake folders so validation passes
    (tmp_path / "model.pt").touch()
    (tmp_path / "data.yaml").touch()
    (tmp_path / "run").mkdir()
    (tmp_path / "images").mkdir()
    
    ctx = ValidationContext(args)
    assert ctx.quantize_out == output_root / "quantized"
    
    # 2. Existing quantized dir (with metadata)
    quant_dir = output_root / "quantized"
    quant_dir.mkdir()
    (quant_dir / "quant_manifest.json").touch()
    
    ctx2 = ValidationContext(args)
    assert ctx2.quantize_out == quant_dir
    
    # 3. Force-quantize makes a timestamped dir
    args.force_quantize = True
    ctx3 = ValidationContext(args)
    assert ctx3.quantize_out.name.startswith("quantized_")
    assert ctx3.quantize_out != quant_dir

def test_state_machine_steps_parsing(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (tmp_path / "model.pt").touch()
    (tmp_path / "data.yaml").touch()
    (tmp_path / "run").mkdir()
    (tmp_path / "images").mkdir()
    
    parser = build_arg_parser()
    
    # 1. Default (all steps)
    args = parser.parse_args([
        "--model", str(tmp_path / "model.pt"),
        "--data", str(tmp_path / "data.yaml"),
        "--run-dir", str(tmp_path / "run"),
        "--images-dir", str(tmp_path / "images"),
        "--output-root", str(output_root),
    ])
    ctx = ValidationContext(args)
    sm = ValidationStateMachine(ctx)
    assert len(sm.get_active_steps()) == 7
    
    # 2. Skip steps
    args.skip_steps = "extract,quantize"
    ctx = ValidationContext(args)
    sm = ValidationStateMachine(ctx)
    active = sm.get_active_steps()
    assert "extract" not in active
    assert "quantize" not in active
    assert "baseline_latency" in active
    
    # 3. Run steps subset
    args.skip_steps = None
    args.run_steps = "quantize,quantized_latency"
    ctx = ValidationContext(args)
    sm = ValidationStateMachine(ctx)
    active = sm.get_active_steps()
    assert active == ["quantize", "quantized_latency"]
    
    # 4. Skip baseline alias
    args.run_steps = None
    args.skip_baseline = True
    ctx = ValidationContext(args)
    sm = ValidationStateMachine(ctx)
    active = sm.get_active_steps()
    assert "extract" not in active
    assert "baseline_threshold" not in active
    assert "baseline_latency" not in active
    assert "baseline_robust" not in active
    assert "quantize" in active

@patch("subprocess.run")
def test_state_machine_execution(mock_run, tmp_path):
    mock_run.return_value = MagicMock(returncode=0)
    
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    (tmp_path / "model.pt").touch()
    (tmp_path / "data.yaml").touch()
    (tmp_path / "run").mkdir()
    (tmp_path / "images").mkdir()
    
    parser = build_arg_parser()
    args = parser.parse_args([
        "--model", str(tmp_path / "model.pt"),
        "--data", str(tmp_path / "data.yaml"),
        "--run-dir", str(tmp_path / "run"),
        "--images-dir", str(tmp_path / "images"),
        "--output-root", str(output_root),
        "--run-steps", "extract"
    ])
    
    ctx = ValidationContext(args)
    sm = ValidationStateMachine(ctx)
    res = sm.execute()
    assert res == 0
    assert mock_run.call_count == 1
