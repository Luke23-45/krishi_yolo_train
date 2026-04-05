"""
yoloml/data/export_yolo.py
----------------------------
Export a canonical Hugging Face dataset into Ultralytics YOLO format.
"""

from __future__ import annotations

from pathlib import Path

from yoloml.config import ExportConfig
from yoloml.data.canonical import export_yolo_from_canonical
from yoloml.pipeline import DataManifest, load_cli_config, parse_stage_args, read_manifest


def main(argv: list[str] | None = None) -> None:
    cli_args, overrides = parse_stage_args("Export a canonical dataset to YOLO format", argv=argv)
    cfg = load_cli_config(overrides=overrides)
    args: ExportConfig = cfg.export

    if cli_args.manifest:
        args.manifest = cli_args.manifest
    if cli_args.output_root:
        args.output = cli_args.output_root

    if args.manifest:
        manifest = read_manifest(args.manifest, DataManifest)
        input_path = Path(manifest.canonical_root).resolve()
        output_path = Path(args.output).resolve() if args.output else Path(manifest.yolo_root).resolve()
    else:
        input_path = Path(args.input).resolve()
        output_path = Path(args.output).resolve()

    stats = export_yolo_from_canonical(input_path, output_path)
    print(
        f"Exported YOLO dataset to {output_path} "
        f"({stats['train_images']} train, {stats['val_images']} val)"
    )


if __name__ == "__main__":
    main()
