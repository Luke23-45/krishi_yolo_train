from __future__ import annotations

import builtins
import os
import sys
import types

import pytest

from yoloml.audit import failure_injection_audit
from yoloml.config import TelemetryConfig, load_config
from yoloml.utils.telemetry import setup_telemetry


def _clear_wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ["WANDB_MODE", "WANDB_DISABLED", "WANDB_PROJECT", "WANDB_API_KEY"]:
        monkeypatch.delenv(key, raising=False)


def test_load_config_defaults_telemetry_mode_online():
    cfg = load_config()
    assert cfg.telemetry.mode == "online"


def test_load_config_rejects_invalid_telemetry_mode():
    with pytest.raises(ValueError, match="Invalid telemetry.mode"):
        load_config(["telemetry.mode=bogus"])


def test_setup_telemetry_disabled_when_enable_wandb_false(monkeypatch: pytest.MonkeyPatch):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_PROJECT", "stale")

    setup_telemetry(TelemetryConfig(project="proj", enable_wandb=False, mode="online"))

    assert os.environ["WANDB_MODE"] == "disabled"
    assert os.environ["WANDB_DISABLED"] == "true"
    assert "WANDB_PROJECT" not in os.environ


def test_setup_telemetry_disabled_mode(monkeypatch: pytest.MonkeyPatch):
    _clear_wandb_env(monkeypatch)

    setup_telemetry(TelemetryConfig(project="proj", enable_wandb=True, mode="disabled"))

    assert os.environ["WANDB_MODE"] == "disabled"
    assert os.environ["WANDB_DISABLED"] == "true"


def test_setup_telemetry_offline_mode(monkeypatch: pytest.MonkeyPatch):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace())

    setup_telemetry(TelemetryConfig(project="proj", run_name="run1", enable_wandb=True, mode="offline"))

    assert os.environ["WANDB_MODE"] == "offline"
    assert "WANDB_DISABLED" not in os.environ
    assert os.environ["WANDB_PROJECT"] == "proj"


def test_setup_telemetry_online_mode(monkeypatch: pytest.MonkeyPatch):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace())

    setup_telemetry(TelemetryConfig(project="proj", run_name="run1", enable_wandb=True, mode="online"))

    assert os.environ["WANDB_MODE"] == "online"
    assert "WANDB_DISABLED" not in os.environ
    assert os.environ["WANDB_PROJECT"] == "proj"


def test_setup_telemetry_import_failure_falls_back_to_disabled_online(monkeypatch: pytest.MonkeyPatch):
    _clear_wandb_env(monkeypatch)
    monkeypatch.setenv("WANDB_PROJECT", "stale")
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("missing wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    setup_telemetry(TelemetryConfig(project="proj", enable_wandb=True, mode="online"))

    assert os.environ["WANDB_MODE"] == "disabled"
    assert os.environ["WANDB_DISABLED"] == "true"
    assert "WANDB_PROJECT" not in os.environ


def test_setup_telemetry_import_failure_falls_back_to_disabled_offline(monkeypatch: pytest.MonkeyPatch):
    _clear_wandb_env(monkeypatch)
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("missing wandb")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    setup_telemetry(TelemetryConfig(project="proj", enable_wandb=True, mode="offline"))

    assert os.environ["WANDB_MODE"] == "disabled"
    assert os.environ["WANDB_DISABLED"] == "true"


def test_failure_injection_audit_requires_wandb_key_only_for_online(monkeypatch: pytest.MonkeyPatch):
    _clear_wandb_env(monkeypatch)

    online_cfg = load_config(["telemetry.enable_wandb=true", "telemetry.mode=online"])
    offline_cfg = load_config(["telemetry.enable_wandb=true", "telemetry.mode=offline"])
    disabled_cfg = load_config(["telemetry.enable_wandb=true", "telemetry.mode=disabled"])
    hard_disabled_cfg = load_config(["telemetry.enable_wandb=false", "telemetry.mode=online"])

    online_checks = failure_injection_audit(online_cfg)
    offline_checks = failure_injection_audit(offline_cfg)
    disabled_checks = failure_injection_audit(disabled_cfg)
    hard_disabled_checks = failure_injection_audit(hard_disabled_cfg)

    assert any(check["name"] == "missing_wandb_key" and check["would_fail"] for check in online_checks["checks"])
    assert any(check["name"] == "missing_wandb_key" and not check["would_fail"] for check in offline_checks["checks"])
    assert any(check["name"] == "missing_wandb_key" and not check["would_fail"] for check in disabled_checks["checks"])
    assert any(check["name"] == "missing_wandb_key" and not check["would_fail"] for check in hard_disabled_checks["checks"])
