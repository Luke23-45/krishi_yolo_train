from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from yoloml.audit import dry_run_audit, static_contract_audit
from yoloml.config import DatasetProvisioningConfig, TrainingConfig, load_config
from yoloml.data.dataset import DatasetManager
from yoloml.data.validate import validate_canonical, validate_yolo
from yoloml.data import export_yolo as export_yolo_module
from yoloml.launch import main as launch_main
from yoloml.models import package as model_package_module
from yoloml.models import quantize as quantize_module
from yoloml.models import validate as model_validate_module
from yoloml.pipeline import (
    DataManifest,
    ModelValidationManifest,
    QuantManifest,
    TrainManifest,
    create_run_context,
    write_manifest,
)
from yoloml.training import train as train_module
from yoloml.training import main as train_main
from yoloml.training.train import train


def test_config_schema_and_audit_contracts():
    cfg = load_config()
    assert "run" in cfg.__dict__
    assert "data_package" in cfg.__dict__
    assert "model_validate" in cfg.__dict__
    assert "package" in cfg.__dict__

    static_report = static_contract_audit(cfg)
    dry_report = dry_run_audit(cfg)
    assert static_report["valid"] is True
    assert dry_report["paths_are_distinct"] is True


def test_dataset_manager_uses_local_yolo_without_mutating_shared_yaml(tmp_path: Path, yolo_dataset: Path):
    output_root = tmp_path / "run_train"
    cfg = DatasetProvisioningConfig(
        yolo_root=str(yolo_dataset),
        canonical_root=str(tmp_path / "missing_canonical"),
        hf_repo_id="",
        allow_fallback=False,
    )
    manager = DatasetManager(cfg)
    prepared = manager.prepare_data(output_root=output_root)

    assert prepared.source == "local-yolo"
    assert prepared.data_yaml == output_root / "data_resolved.yaml"
    assert prepared.data_yaml.exists()
    assert (yolo_dataset / "data.yaml").exists()


def test_dataset_manager_exports_from_canonical(tmp_path: Path, canonical_dataset: Path):
    yolo_root = tmp_path / "exported_yolo"
    cfg = DatasetProvisioningConfig(
        yolo_root=str(yolo_root),
        canonical_root=str(canonical_dataset),
        hf_repo_id="",
        allow_fallback=False,
    )
    manager = DatasetManager(cfg)
    prepared = manager.prepare_data(output_root=tmp_path / "run_data")

    assert prepared.source == "local-canonical-export"
    assert (yolo_root / "data.yaml").exists()
    assert prepared.data_yaml.exists()


def test_dataset_manager_falls_back_to_materialization(tmp_path: Path, monkeypatch):
    yolo_root = tmp_path / "materialized_yolo"
    cfg = DatasetProvisioningConfig(
        yolo_root=str(yolo_root),
        canonical_root=str(tmp_path / "canonical"),
        hf_repo_id="repo",
        allow_fallback=True,
    )
    manager = DatasetManager(cfg)

    monkeypatch.setattr(manager, "_sync_from_huggingface", lambda: False)

    def fake_materialize():
        for split in ("train", "val"):
            (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (yolo_root / "images" / split / f"{split}.jpg").write_bytes(b"img")
            (yolo_root / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        (yolo_root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: leaf\n", encoding="utf-8")
        return True

    monkeypatch.setattr(manager, "_trigger_materialization", fake_materialize)
    prepared = manager.prepare_data(output_root=tmp_path / "run_materialized")
    assert prepared.source == "materialized"
    assert prepared.data_yaml.exists()


def test_dataset_manager_hub_sync_supports_canonical_repo(tmp_path: Path, monkeypatch):
    canonical_root = tmp_path / "canonical"
    yolo_root = tmp_path / "yolo"
    cfg = DatasetProvisioningConfig(
        yolo_root=str(yolo_root),
        canonical_root=str(canonical_root),
        hf_repo_id="repo",
        allow_fallback=False,
    )
    manager = DatasetManager(cfg)

    def fake_snapshot_download(repo_id, repo_type, local_dir, local_dir_use_symlinks, cache_dir, resume_download):
        target = Path(local_dir)
        if target == canonical_root:
            (target / "train" / "images").mkdir(parents=True, exist_ok=True)
            (target / "val" / "images").mkdir(parents=True, exist_ok=True)
            (target / "parquet").mkdir(parents=True, exist_ok=True)
            (target / "classes.json").write_text('{"nc": 1, "names": {"0": "leaf"}}', encoding="utf-8")
            (target / "train" / "metadata.jsonl").write_text("{}", encoding="utf-8")
            (target / "val" / "metadata.jsonl").write_text("{}", encoding="utf-8")
            (target / "parquet" / "train_metadata.parquet").write_text("fixture", encoding="utf-8")
            (target / "parquet" / "val_metadata.parquet").write_text("fixture", encoding="utf-8")
        else:
            target.mkdir(parents=True, exist_ok=True)

    def fake_export_yolo(canonical, yolo):
        for split in ("train", "val"):
            (yolo / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo / "labels" / split).mkdir(parents=True, exist_ok=True)
            (yolo / "images" / split / f"{split}.jpg").write_bytes(b"img")
            (yolo / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        (yolo / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: leaf\n", encoding="utf-8")

    fake_hf_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)
    monkeypatch.setattr("yoloml.data.dataset.export_yolo_from_canonical", fake_export_yolo)
    prepared = manager.prepare_data(output_root=tmp_path / "run_hub")
    assert prepared.source == "hub-canonical-export"
    assert (yolo_root / "data.yaml").exists()


def test_dataset_manager_accepts_data_yaml_with_explicit_path_root(tmp_path: Path):
    actual_root = tmp_path / "actual_yolo"
    config_root = tmp_path / "config_only"
    for split in ("train", "val"):
        (actual_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (actual_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (actual_root / "images" / split / f"{split}.jpg").write_bytes(b"img")
        (actual_root / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "data.yaml").write_text(
        yaml.dump({
            "path": str(actual_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": {0: "leaf"},
        }, sort_keys=False),
        encoding="utf-8",
    )

    manager = DatasetManager(DatasetProvisioningConfig(
        yolo_root=str(config_root),
        canonical_root=str(tmp_path / "missing_canonical"),
        hf_repo_id="",
        allow_fallback=False,
    ))
    prepared = manager.prepare_data(output_root=tmp_path / "run_explicit_path")
    assert prepared.source == "local-yolo"
    assert prepared.data_yaml.exists()


def test_train_writes_manifest_and_rejects_invalid_data(tmp_path: Path, yolo_dataset: Path):
    output_root = tmp_path / "train"
    output_root.mkdir()
    snapshot = output_root / "config.json"
    snapshot.write_text("{}", encoding="utf-8")

    manifest = train(
        TrainingConfig(dry_run=True),
        yolo_dataset / "data.yaml",
        telemetry_cfg=load_config().telemetry,
        output_root=output_root,
        training_config_snapshot=snapshot,
        run_id="train_test",
    )
    assert manifest.valid is True
    assert manifest.balance_report.endswith("balance_report.json")

    with pytest.raises(FileNotFoundError):
        train(
            TrainingConfig(dry_run=True),
            tmp_path / "missing.yaml",
            telemetry_cfg=load_config().telemetry,
            output_root=output_root,
            training_config_snapshot=snapshot,
            run_id="train_test",
        )


def test_train_main_accepts_documented_cli_flags(tmp_path: Path, yolo_dataset: Path):
    run_id = "cli_test"
    output_root = tmp_path / "train_cli"
    train_main([
        "run.run_id=" + run_id,
        "run.root_dir=" + str(output_root),
        "training.data=" + str((yolo_dataset / "data.yaml").resolve()),
        "training.epochs=3",
        "training.batch=2",
        "training.imgsz=320",
        "training.dry_run=true",
        "telemetry.mode=disabled",
    ])

    manifest_path = output_root / run_id / "train" / "train_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["valid"] is True
    assert Path(manifest["verified_data_yaml"]).exists()
    assert manifest["output_root"] == str((output_root / run_id / "train").resolve())


def test_quantize_consumes_train_manifest_and_missing_weights_fail(tmp_path: Path, monkeypatch, yolo_dataset: Path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"pt")
    train_manifest = TrainManifest(
        run_id="run123",
        data_manifest=None,
        verified_data_yaml=str((yolo_dataset / "data.yaml").resolve()),
        training_config_snapshot=str((tmp_path / "config.json").resolve()),
        output_root=str((tmp_path / "train").resolve()),
        balance_report=str((tmp_path / "balance_report.json").resolve()),
        telemetry_project="proj",
        telemetry_run_name="run123",
        best_weights=str(weights.resolve()),
        last_weights=None,
        valid=True,
    )
    train_manifest_path = tmp_path / "train_manifest.json"
    write_manifest(train_manifest_path, train_manifest)

    def fake_export(model_path, level, data_yaml, imgsz, output_dir):
        out = output_dir / level / f"{level}.tflite"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"tflite")
        meta = out.parent / "metadata.json"
        meta.write_text("{}", encoding="utf-8")
        return {
            "level": level,
            "success": True,
            "tflite_path": str(out.resolve()),
            "metadata_path": str(meta.resolve()),
            "model_size_bytes": 10,
            "model_size_mb": 0.01,
            "pt_size_bytes": 20,
            "pt_size_mb": 0.02,
            "compression_ratio": 2.0,
            "export_time_seconds": 0.1,
            "imgsz": imgsz,
        }

    monkeypatch.setattr(quantize_module, "export_tflite", fake_export)
    monkeypatch.setattr(quantize_module, "generate_comparison", lambda results, output_dir: output_dir / "comparison_report.json")
    quantize_module.main(["--manifest", str(train_manifest_path), "--output-root", str(tmp_path / "quant")])
    quant_manifest = json.loads((tmp_path / "quant" / "quant_manifest.json").read_text(encoding="utf-8"))
    assert quant_manifest["valid"] is True
    assert len(quant_manifest["levels"]) == 3

    train_manifest.best_weights = str((tmp_path / "missing.pt").resolve())
    write_manifest(train_manifest_path, train_manifest)
    with pytest.raises(FileNotFoundError):
        quantize_module.main(["--manifest", str(train_manifest_path), "--output-root", str(tmp_path / "quant_fail")])


def test_model_validation_and_packaging(tmp_path: Path, monkeypatch):
    tflite_path = tmp_path / "quant" / "fp32" / "model.tflite"
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(b"tflite")

    quant_manifest = QuantManifest(
        run_id="run123",
        train_manifest=None,
        source_model=str((tmp_path / "best.pt").resolve()),
        data_yaml=None,
        output_root=str((tmp_path / "quant").resolve()),
        levels=[{"level": "fp32", "success": True, "tflite_path": str(tflite_path.resolve())}],
        comparison_report=None,
        valid=True,
    )
    quant_manifest_path = tmp_path / "quant_manifest.json"
    write_manifest(quant_manifest_path, quant_manifest)

    monkeypatch.setattr(model_validate_module, "validate_tflite", lambda path: {"valid": True, "input_shape": [[1, 640, 640, 3]], "output_shape": [[1, 10]], "input_dtype": ["float32"], "output_dtype": ["float32"]})
    model_validate_module.main(["--manifest", str(quant_manifest_path), "--output-root", str(tmp_path / "model_validate")])
    validation_manifest_path = tmp_path / "model_validate" / "model_validation_manifest.json"
    validation_manifest = json.loads(validation_manifest_path.read_text(encoding="utf-8"))
    assert validation_manifest["valid"] is True

    model_package_module.main(["--manifest", str(validation_manifest_path), "--output-root", str(tmp_path / "package")])
    package_manifest = json.loads((tmp_path / "package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert package_manifest["valid"] is True
    assert Path(package_manifest["packaged_bundle"]).exists()


def test_model_package_publish_result_is_recorded(tmp_path: Path, monkeypatch):
    tflite_path = tmp_path / "quant" / "fp32" / "model.tflite"
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(b"tflite")

    quant_manifest = QuantManifest(
        run_id="run123",
        train_manifest=None,
        source_model=str((tmp_path / "best.pt").resolve()),
        data_yaml=None,
        output_root=str((tmp_path / "quant").resolve()),
        levels=[{"level": "fp32", "success": True, "tflite_path": str(tflite_path.resolve())}],
        comparison_report=None,
        valid=True,
    )
    quant_manifest_path = tmp_path / "quant_manifest.json"
    write_manifest(quant_manifest_path, quant_manifest)

    validation_manifest = ModelValidationManifest(
        run_id="run123",
        quant_manifest=str(quant_manifest_path.resolve()),
        output_root=str((tmp_path / "model_validate").resolve()),
        validated_artifacts=[{"level": "fp32", "valid": True, "tflite_path": str(tflite_path.resolve())}],
        valid=True,
    )
    validation_manifest_path = tmp_path / "model_validation_manifest.json"
    write_manifest(validation_manifest_path, validation_manifest)

    class FakeHfApi:
        def create_repo(self, repo_id, private, exist_ok, repo_type):
            assert repo_id == "hellxhell/krishi-bouncer-model"
            assert repo_type == "model"

        def upload_folder(self, folder_path, repo_id, repo_type, path_in_repo, commit_message):
            assert Path(folder_path).exists()
            assert repo_id == "hellxhell/krishi-bouncer-model"
            assert repo_type == "model"
            return types.SimpleNamespace(oid="commit123")

    fake_hf_module = types.SimpleNamespace(HfApi=FakeHfApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)

    model_package_module.main([
        "--manifest", str(validation_manifest_path),
        "--output-root", str(tmp_path / "package"),
        "package.repo_id=hellxhell/krishi-bouncer-model",
    ])
    package_manifest = json.loads((tmp_path / "package" / "package_manifest.json").read_text(encoding="utf-8"))
    assert package_manifest["publish_result"]["repo_id"] == "hellxhell/krishi-bouncer-model"
    assert package_manifest["publish_result"]["commit"] == "commit123"


def test_data_package_config_regression(tmp_path: Path, canonical_dataset: Path, monkeypatch):
    from yoloml.data import package as data_package_module

    manifest = DataManifest(
        run_id="run123",
        canonical_root=str(canonical_dataset.resolve()),
        yolo_root=str((tmp_path / "yolo").resolve()),
        canonical_validation_report=str((canonical_dataset / "validation_report.json").resolve()),
        yolo_validation_report=str((tmp_path / "yolo" / "validation_report.json").resolve()),
        materialization_report=str((canonical_dataset / "materialization_report.json").resolve()),
        schema_hash="hash",
        class_names={"0": "leaf"},
        valid=True,
    )
    manifest_path = tmp_path / "data_manifest.json"
    write_manifest(manifest_path, manifest)

    monkeypatch.setattr(data_package_module, "export_webdataset_from_canonical", lambda dataset_root, output_root, shard_size_mb: {"split_counts": {"train": 1, "val": 1}, "shard_counts": {"train": 1, "val": 1}, "shard_size_mb": shard_size_mb})
    data_package_module.main(["--manifest", str(manifest_path), "--output-root", str(tmp_path / "pkgout"), "data_package.archive_format=zip"])
    assert (tmp_path / "pkgout" / f"{canonical_dataset.name}.zip").exists()


def test_validation_output_root_is_honored(tmp_path: Path, canonical_dataset: Path, yolo_dataset: Path):
    canonical_report = validate_canonical(canonical_dataset, report_path=tmp_path / "reports" / "canonical.json")
    yolo_report = validate_yolo(yolo_dataset, report_path=tmp_path / "reports" / "yolo.json")
    assert canonical_report == tmp_path / "reports" / "canonical.json"
    assert yolo_report == tmp_path / "reports" / "yolo.json"
    assert canonical_report.exists()
    assert yolo_report.exists()


def test_export_yolo_honors_output_root_with_manifest(tmp_path: Path, canonical_dataset: Path, monkeypatch):
    manifest = DataManifest(
        run_id="run123",
        canonical_root=str(canonical_dataset.resolve()),
        yolo_root=str((tmp_path / "default_yolo").resolve()),
        canonical_validation_report=str((tmp_path / "canonical_validation_report.json").resolve()),
        yolo_validation_report=str((tmp_path / "yolo_validation_report.json").resolve()),
        materialization_report=str((canonical_dataset / "materialization_report.json").resolve()),
        schema_hash="hash",
        class_names={"0": "leaf"},
        valid=True,
    )
    manifest_path = tmp_path / "data_manifest.json"
    write_manifest(manifest_path, manifest)

    captured = {}

    def fake_export(input_path, output_path):
        captured["input"] = input_path
        captured["output"] = output_path
        return {"train_images": 1, "val_images": 1}

    monkeypatch.setattr(export_yolo_module, "export_yolo_from_canonical", fake_export)
    export_yolo_module.main([
        "--manifest", str(manifest_path),
        "--output-root", str(tmp_path / "override_yolo"),
    ])

    assert captured["input"] == canonical_dataset.resolve()
    assert captured["output"] == (tmp_path / "override_yolo").resolve()


def test_validate_yolo_honors_data_yaml_path_field(tmp_path: Path):
    actual_root = tmp_path / "actual_yolo"
    config_root = tmp_path / "config_only"
    for split in ("train", "val"):
        (actual_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (actual_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (actual_root / "images" / split / f"{split}.jpg").write_bytes(b"img")
        (actual_root / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "data.yaml").write_text(
        yaml.dump({
            "path": str(actual_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": {0: "leaf"},
        }, sort_keys=False),
        encoding="utf-8",
    )

    report = validate_yolo(config_root, report_path=tmp_path / "reports" / "yolo_explicit_path.json")
    assert report.exists()


def test_launch_full_smoke(tmp_path: Path, monkeypatch):
    def fake_prepare_data(self, output_root=None):
        canonical_root = tmp_path / "hf_dataset"
        yolo_root = tmp_path / "krishi_bouncer_dataset"
        (canonical_root / "train" / "images").mkdir(parents=True, exist_ok=True)
        (canonical_root / "val" / "images").mkdir(parents=True, exist_ok=True)
        (canonical_root / "parquet").mkdir(parents=True, exist_ok=True)
        (canonical_root / "classes.json").write_text('{"nc": 1, "names": {"0": "leaf"}}', encoding="utf-8")
        (canonical_root / "licenses.json").write_text("[]", encoding="utf-8")
        (canonical_root / "README.md").write_text("fixture", encoding="utf-8")
        (canonical_root / "train" / "metadata.jsonl").write_text("{}", encoding="utf-8")
        (canonical_root / "val" / "metadata.jsonl").write_text("{}", encoding="utf-8")
        (canonical_root / "parquet" / "train_metadata.parquet").write_text("fixture", encoding="utf-8")
        (canonical_root / "parquet" / "val_metadata.parquet").write_text("fixture", encoding="utf-8")
        (canonical_root / "materialization_report.json").write_text('{"curation": {}}', encoding="utf-8")
        for split in ("train", "val"):
            (yolo_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (yolo_root / "labels" / split).mkdir(parents=True, exist_ok=True)
            (yolo_root / "images" / split / f"{split}.jpg").write_bytes(b"img")
            (yolo_root / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
        (yolo_root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: leaf\n", encoding="utf-8")
        return type("Prepared", (), {
            "data_yaml": (output_root / "data_resolved.yaml") if output_root else (yolo_root / "data.yaml"),
            "yolo_root": yolo_root,
            "canonical_root": canonical_root,
            "source": "hub-canonical-export",
        })()

    monkeypatch.setattr("yoloml.launch.DatasetManager.prepare_data", fake_prepare_data)
    monkeypatch.setattr("yoloml.launch.read_schema_names", lambda path: {0: "leaf"})
    monkeypatch.setattr("yoloml.launch.validate_canonical", lambda path, do_check_images=False, report_path=None: (report_path or (path / "validation_report.json")))
    monkeypatch.setattr("yoloml.launch.validate_yolo", lambda path, do_check_images=False, report_path=None: (report_path or (path / "validation_report.json")))

    def fake_train_main(argv):
        output_root = Path(argv[argv.index("--output-root") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = TrainManifest(
            run_id="smoke",
            data_manifest=argv[argv.index("--manifest") + 1],
            verified_data_yaml=str((tmp_path / "krishi_bouncer_dataset" / "data.yaml").resolve()),
            training_config_snapshot=str((output_root / "resolved_config.json").resolve()),
            output_root=str(output_root.resolve()),
            balance_report=str((output_root / "balance_report.json").resolve()),
            telemetry_project="proj",
            telemetry_run_name="smoke",
            best_weights=str((output_root / "best.pt").resolve()),
            last_weights=str((output_root / "last.pt").resolve()),
            valid=True,
        )
        Path(manifest.best_weights).write_bytes(b"pt")
        Path(manifest.last_weights).write_bytes(b"pt")
        write_manifest(output_root / "train_manifest.json", manifest)

    def fake_quant_main(argv):
        output_root = Path(argv[argv.index("--output-root") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        tflite = output_root / "fp32.tflite"
        tflite.write_bytes(b"tflite")
        manifest = QuantManifest(
            run_id="smoke",
            train_manifest=str((tmp_path / "outputs" / "runs" / "smoke" / "train" / "train_manifest.json").resolve()),
            source_model=str((tmp_path / "outputs" / "runs" / "smoke" / "train" / "best.pt").resolve()),
            data_yaml=str((tmp_path / "krishi_bouncer_dataset" / "data.yaml").resolve()),
            output_root=str(output_root.resolve()),
            levels=[{"level": "fp32", "success": True, "tflite_path": str(tflite.resolve())}],
            comparison_report=None,
            valid=True,
        )
        write_manifest(output_root / "quant_manifest.json", manifest)

    def fake_model_validate_main(argv):
        output_root = Path(argv[argv.index("--output-root") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        manifest = ModelValidationManifest(
            run_id="smoke",
            quant_manifest=str((tmp_path / "outputs" / "runs" / "smoke" / "quantize" / "quant_manifest.json").resolve()),
            output_root=str(output_root.resolve()),
            validated_artifacts=[{"level": "fp32", "valid": True, "tflite_path": str((tmp_path / "outputs" / "runs" / "smoke" / "quantize" / "fp32.tflite").resolve())}],
            valid=True,
        )
        write_manifest(output_root / "model_validation_manifest.json", manifest)

    def fake_package_main(argv):
        output_root = Path(argv[argv.index("--output-root") + 1])
        output_root.mkdir(parents=True, exist_ok=True)
        bundle = output_root / "bundle.zip"
        bundle.write_bytes(b"zip")
        write_manifest(output_root / "package_manifest.json", {
            "run_id": "smoke",
            "model_validation_manifest": argv[argv.index("--manifest") + 1],
            "output_root": str(output_root.resolve()),
            "packaged_bundle": str(bundle.resolve()),
            "included_artifacts": [],
            "metadata": {},
            "publish_result": None,
            "valid": True,
        })

    monkeypatch.setattr("yoloml.launch.train_module.main", fake_train_main)
    monkeypatch.setattr("yoloml.launch.quantize_module.main", fake_quant_main)
    monkeypatch.setattr("yoloml.launch.model_validate_module.main", fake_model_validate_main)
    monkeypatch.setattr("yoloml.launch.model_package_module.main", fake_package_main)

    launch_main([
        "--mode", "full",
        "--run-id", "smoke",
        "materialize.canonical_root=" + str((tmp_path / "hf_dataset").resolve()),
        "materialize.yolo_root=" + str((tmp_path / "krishi_bouncer_dataset").resolve()),
        "run.root_dir=" + str((tmp_path / "outputs" / "runs").resolve()),
    ])

    run_context = create_run_context(load_config(overrides=[
        "run.run_id=smoke",
        "run.root_dir=" + str((tmp_path / "outputs" / "runs").resolve()),
    ]), run_id="smoke")
    assert (run_context.data_dir / "data_manifest.json").exists()
    assert (run_context.train_dir / "train_manifest.json").exists()
    assert (run_context.quantize_dir / "quant_manifest.json").exists()
    assert (run_context.model_validation_dir / "model_validation_manifest.json").exists()
    assert (run_context.package_dir / "package_manifest.json").exists()


def test_launch_resume_reuses_valid_manifests(tmp_path: Path, monkeypatch):
    cfg = load_config(overrides=[
        "run.run_id=resume_smoke",
        "run.root_dir=" + str((tmp_path / "outputs" / "runs").resolve()),
        "run.resume=true",
    ])
    run_context = create_run_context(cfg, run_id="resume_smoke")
    for stage_dir in [
        run_context.data_dir,
        run_context.train_dir,
        run_context.quantize_dir,
        run_context.model_validation_dir,
        run_context.package_dir,
    ]:
        stage_dir.mkdir(parents=True, exist_ok=True)

    data_manifest = DataManifest(
        run_id="resume_smoke",
        canonical_root=str((tmp_path / "hf_dataset").resolve()),
        yolo_root=str((tmp_path / "krishi_bouncer_dataset").resolve()),
        canonical_validation_report=str((run_context.data_dir / "canonical_validation_report.json").resolve()),
        yolo_validation_report=str((run_context.data_dir / "yolo_validation_report.json").resolve()),
        materialization_report=str((tmp_path / "hf_dataset" / "materialization_report.json").resolve()),
        schema_hash="hash",
        class_names={"0": "leaf"},
        valid=True,
    )
    train_manifest = TrainManifest(
        run_id="resume_smoke",
        data_manifest=str((run_context.data_dir / "data_manifest.json").resolve()),
        verified_data_yaml=str((tmp_path / "krishi_bouncer_dataset" / "data.yaml").resolve()),
        training_config_snapshot=str((run_context.train_dir / "resolved_config.json").resolve()),
        output_root=str(run_context.train_dir.resolve()),
        balance_report=str((run_context.train_dir / "balance_report.json").resolve()),
        telemetry_project="proj",
        telemetry_run_name="resume_smoke",
        best_weights=str((run_context.train_dir / "best.pt").resolve()),
        last_weights=None,
        valid=True,
    )
    quant_manifest = QuantManifest(
        run_id="resume_smoke",
        train_manifest=str((run_context.train_dir / "train_manifest.json").resolve()),
        source_model=str((run_context.train_dir / "best.pt").resolve()),
        data_yaml=str((tmp_path / "krishi_bouncer_dataset" / "data.yaml").resolve()),
        output_root=str(run_context.quantize_dir.resolve()),
        levels=[{
            "level": "fp32",
            "success": True,
            "tflite_path": str((run_context.quantize_dir / "fp32.tflite").resolve()),
            "metadata_path": str((run_context.quantize_dir / "fp32_metadata.json").resolve()),
        }],
        comparison_report=None,
        valid=True,
    )
    validation_manifest = ModelValidationManifest(
        run_id="resume_smoke",
        quant_manifest=str((run_context.quantize_dir / "quant_manifest.json").resolve()),
        output_root=str(run_context.model_validation_dir.resolve()),
        validated_artifacts=[{"level": "fp32", "valid": True, "tflite_path": str((run_context.quantize_dir / "fp32.tflite").resolve())}],
        valid=True,
    )
    package_manifest = {
        "run_id": "resume_smoke",
        "model_validation_manifest": str((run_context.model_validation_dir / "model_validation_manifest.json").resolve()),
        "output_root": str(run_context.package_dir.resolve()),
        "packaged_bundle": str((run_context.package_dir / "bundle.zip").resolve()),
        "included_artifacts": [],
        "metadata": {},
        "publish_result": None,
        "valid": True,
    }

    Path(data_manifest.materialization_report).parent.mkdir(parents=True, exist_ok=True)
    Path(data_manifest.canonical_root).mkdir(parents=True, exist_ok=True)
    Path(data_manifest.yolo_root).mkdir(parents=True, exist_ok=True)
    Path(data_manifest.materialization_report).write_text("{}", encoding="utf-8")
    Path(data_manifest.canonical_validation_report).write_text("{}", encoding="utf-8")
    Path(data_manifest.yolo_validation_report).write_text("{}", encoding="utf-8")
    Path(train_manifest.verified_data_yaml).parent.mkdir(parents=True, exist_ok=True)
    Path(train_manifest.verified_data_yaml).write_text("path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: leaf\n", encoding="utf-8")
    Path(train_manifest.training_config_snapshot).write_text("{}", encoding="utf-8")
    Path(train_manifest.balance_report).write_text("{}", encoding="utf-8")
    Path(train_manifest.best_weights).write_bytes(b"pt")
    Path(quant_manifest.levels[0]["tflite_path"]).write_bytes(b"tflite")
    Path(quant_manifest.levels[0]["metadata_path"]).write_text("{}", encoding="utf-8")
    Path(package_manifest["packaged_bundle"]).write_bytes(b"zip")

    write_manifest(run_context.data_dir / "data_manifest.json", data_manifest)
    write_manifest(run_context.train_dir / "train_manifest.json", train_manifest)
    write_manifest(run_context.quantize_dir / "quant_manifest.json", quant_manifest)
    write_manifest(run_context.model_validation_dir / "model_validation_manifest.json", validation_manifest)
    write_manifest(run_context.package_dir / "package_manifest.json", package_manifest)

    monkeypatch.setattr("yoloml.launch.DatasetManager.prepare_data", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("data stage should be resumed")))
    monkeypatch.setattr("yoloml.launch.train_module.main", lambda argv: (_ for _ in ()).throw(AssertionError("train stage should be resumed")))
    monkeypatch.setattr("yoloml.launch.quantize_module.main", lambda argv: (_ for _ in ()).throw(AssertionError("quant stage should be resumed")))
    monkeypatch.setattr("yoloml.launch.model_validate_module.main", lambda argv: (_ for _ in ()).throw(AssertionError("model validation stage should be resumed")))
    monkeypatch.setattr("yoloml.launch.model_package_module.main", lambda argv: (_ for _ in ()).throw(AssertionError("package stage should be resumed")))

    launch_main([
        "--mode", "full",
        "--run-id", "resume_smoke",
        "run.root_dir=" + str((tmp_path / "outputs" / "runs").resolve()),
        "run.resume=true",
    ])


def test_train_uses_output_root_for_ultralytics_artifacts(tmp_path: Path, yolo_dataset: Path, monkeypatch):
    output_root = tmp_path / "train_artifacts"
    output_root.mkdir()
    snapshot = output_root / "config.json"
    snapshot.write_text("{}", encoding="utf-8")

    class FakeYOLO:
        def __init__(self, model_name):
            self.model_name = model_name
            self.model = types.SimpleNamespace(model=[None, types.SimpleNamespace()])

        def train(self, **kwargs):
            assert kwargs["project"] == str(output_root)
            assert kwargs["name"] == "artifacts"
            save_dir = output_root / "artifacts"
            (save_dir / "weights").mkdir(parents=True, exist_ok=True)
            (save_dir / "weights" / "best.pt").write_bytes(b"pt")
            return types.SimpleNamespace(save_dir=str(save_dir))

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))

    manifest = train(
        TrainingConfig(dry_run=False, no_class_weights=True),
        yolo_dataset / "data.yaml",
        telemetry_cfg=load_config(overrides=["telemetry.mode=disabled"]).telemetry,
        output_root=output_root,
        training_config_snapshot=snapshot,
        run_id="artifact_test",
    )
    assert manifest.valid is True
    assert manifest.best_weights == str((output_root / "artifacts" / "weights" / "best.pt").resolve())
