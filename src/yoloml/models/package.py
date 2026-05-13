"""
yoloml/models/package.py
-------------------------
Package validated model artifacts into a reproducible bundle.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from yoloml.config import ModelPackageConfig
from yoloml.pipeline import (
    ModelValidationManifest,
    PackageManifest,
    QuantManifest,
    TrainManifest,
    create_run_context,
    ensure_stage_dir,
    load_cli_config,
    parse_stage_args,
    read_manifest,
    snapshot_config,
    write_manifest,
)


def _copy_if_exists(src: Path, dst: Path, included: list[str]) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        included.append(str(dst.resolve()))


def _publish_bundle(bundle_root: Path, repo_id: str, private: bool) -> dict[str, str]:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to publish model packages") from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True, repo_type="model")
    commit_info = api.upload_folder(
        folder_path=str(bundle_root),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=".",
        commit_message="Upload model package bundle",
    )
    return {
        "repo_id": repo_id,
        "repo_type": "model",
        "url": f"https://huggingface.co/{repo_id}",
        # CommitInfo object has an 'oid' attribute containing the exact commit hash
        "commit": commit_info.oid, 
    }


def main(argv: list[str] | None = None) -> None:
    cli_args, overrides = parse_stage_args(
        "Package validated model artifacts", 
        argv=argv,
        extra_arguments=[
            (("--package-name",), {"type": str, "default": None}),
            (("--include-reports",), {"action": "store_true"}),
            (("--include-source-weights",), {"action": "store_true"}),
            (("--repo-id",), {"type": str, "default": None}),
            (("--private",), {"action": "store_true"}),
            (("--archive-format",), {"type": str, "default": None}),
        ],
    )
    cfg = load_cli_config(overrides=overrides)
    if cli_args.run_id:
        cfg.run.run_id = cli_args.run_id
    if cli_args.manifest:
        cfg.model_package.manifest = cli_args.manifest
    if cli_args.output_root:
        cfg.model_package.output_root = cli_args.output_root
        
    # Map the custom arguments to the config
    if cli_args.package_name:
        cfg.model_package.package_name = cli_args.package_name
    if cli_args.include_reports:
        cfg.model_package.include_reports = True
    if cli_args.include_source_weights:
        cfg.model_package.include_source_weights = True
    if cli_args.repo_id:
        cfg.model_package.repo_id = cli_args.repo_id
    if cli_args.private:
        cfg.model_package.private = True
    if cli_args.archive_format:
        cfg.model_package.archive_format = cli_args.archive_format

    args: ModelPackageConfig = cfg.model_package
    if not args.manifest:
        raise ValueError("Model packaging requires --manifest pointing to model_validation_manifest.json")

    run_context = create_run_context(cfg, run_id=cfg.run.run_id)
    output_root = (
        ensure_stage_dir(Path(args.output_root).resolve())
        if args.output_root
        else ensure_stage_dir(run_context.package_dir)
    )
    snapshot_config(cfg, output_root / "resolved_config.json")

    model_validation_path = Path(args.manifest).resolve()
    model_validation = read_manifest(model_validation_path, ModelValidationManifest)
    if not model_validation.valid:
        raise RuntimeError(f"Model validation manifest is not valid: {model_validation_path}")

    quant_manifest = read_manifest(model_validation.quant_manifest, QuantManifest)
    bundle_root = ensure_stage_dir(output_root / args.package_name)
    included_artifacts: list[str] =[]

    for artifact in model_validation.validated_artifacts:
        tflite_path = artifact.get("tflite_path")
        if artifact.get("valid") and tflite_path:
            src = Path(tflite_path)
            _copy_if_exists(src, bundle_root / "artifacts" / src.name, included_artifacts)

    if args.include_reports:
        _copy_if_exists(Path(model_validation.quant_manifest), bundle_root / "manifests" / "quant_manifest.json", included_artifacts)
        _copy_if_exists(model_validation_path, bundle_root / "manifests" / "model_validation_manifest.json", included_artifacts)
        if quant_manifest.comparison_report:
            _copy_if_exists(Path(quant_manifest.comparison_report), bundle_root / "reports" / Path(quant_manifest.comparison_report).name, included_artifacts)
        for level in quant_manifest.levels:
            if level.get("metadata_path"):
                meta_path = Path(level["metadata_path"])
                _copy_if_exists(meta_path, bundle_root / "reports" / level["level"] / meta_path.name, included_artifacts)

    if args.include_source_weights and quant_manifest.train_manifest:
        train_manifest = read_manifest(quant_manifest.train_manifest, TrainManifest)
        for candidate in [train_manifest.best_weights, train_manifest.last_weights]:
            if candidate:
                src = Path(candidate)
                _copy_if_exists(src, bundle_root / "source_weights" / src.name, included_artifacts)
        _copy_if_exists(Path(quant_manifest.train_manifest), bundle_root / "manifests" / "train_manifest.json", included_artifacts)

    publish_result = _publish_bundle(bundle_root, args.repo_id, args.private) if args.repo_id else None

    archive_base = output_root / args.package_name
    archive_path = shutil.make_archive(str(archive_base), args.archive_format, root_dir=bundle_root.parent, base_dir=bundle_root.name)
    manifest = PackageManifest(
        run_id=run_context.run_id,
        model_validation_manifest=str(model_validation_path),
        output_root=str(output_root.resolve()),
        packaged_bundle=str(Path(archive_path).resolve()),
        included_artifacts=included_artifacts,
        metadata={
            "archive_format": args.archive_format,
            "artifact_count": len(included_artifacts),
            "repo_id": args.repo_id,
            "private": args.private,
        },
        publish_result=publish_result,
        valid=Path(archive_path).exists(),
    )
    write_manifest(output_root / "package_manifest.json", manifest)


if __name__ == "__main__":
    main()