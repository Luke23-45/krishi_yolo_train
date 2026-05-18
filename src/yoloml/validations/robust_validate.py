"""
yoloml/validations/robust_validate.py
-------------------------------------
Run detailed validation against a YOLO-format validation split and export
prediction-level CSVs for manual review.

Produces (inside <output_dir>/):
    summary.json               - aggregate metrics (P, R, F1, counts, timing)
    summary.csv                - single-row CSV of the same (for thesis tables)
    predictions.csv            - every prediction with match status
    errors.csv                 - TP/FP/FN error catalogue
    per_image_summary.csv      - per-image detection counts and timing
    per_class_metrics.csv      - per-class P, R, F1 breakdown
    confusion_matrix.csv       - class-level confusion matrix
    preview/                   - rendered prediction images (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class GroundTruth:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class Prediction:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def _resolve_split_dir(data_root: Path, split_value: str) -> Path:
    split_path = Path(split_value)
    if split_path.is_absolute():
        return split_path.resolve()
    return (data_root / split_path).resolve()


def load_data_yaml(data_yaml: Path, split: str) -> tuple[Path, Path, dict[int, str]]:
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root_value = cfg.get("path")
    if root_value:
        root_path = Path(root_value)
        data_root = root_path.resolve() if root_path.is_absolute() else (data_yaml.parent / root_path).resolve()
    else:
        data_root = data_yaml.parent.resolve()

    split_key = split
    if split_key not in cfg and split == "val":
        for alias in ("valid", "validation"):
            if alias in cfg:
                split_key = alias
                break

    if split_key not in cfg:
        raise KeyError(f"Split '{split}' not found in {data_yaml}")

    images_dir = _resolve_split_dir(data_root, str(cfg[split_key]))
    parts = list(images_dir.parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        labels_dir = Path(*parts).resolve()
    else:
        labels_dir = (images_dir.parent.parent / "labels" / images_dir.name).resolve()

    raw_names = cfg.get("names", {})
    if isinstance(raw_names, list):
        class_names = {idx: str(name) for idx, name in enumerate(raw_names)}
    else:
        class_names = {int(key): str(value) for key, value in raw_names.items()}

    return images_dir, labels_dir, class_names


def collect_images(images_dir: Path) -> list[Path]:
    return sorted(path.resolve() for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def xywhn_to_xyxy(class_id: int, cx: float, cy: float, w: float, h: float, image_w: int, image_h: int) -> GroundTruth:
    x1 = (cx - w / 2.0) * image_w
    y1 = (cy - h / 2.0) * image_h
    x2 = (cx + w / 2.0) * image_w
    y2 = (cy + h / 2.0) * image_h
    return GroundTruth(class_id=class_id, x1=x1, y1=y1, x2=x2, y2=y2)


def load_ground_truth(label_path: Path, image_w: int, image_h: int) -> list[GroundTruth]:
    if not label_path.exists():
        return []

    boxes: list[GroundTruth] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO label line in {label_path}: {line!r}")
        class_id = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        boxes.append(xywhn_to_xyxy(class_id, cx, cy, w, h, image_w, image_h))
    return boxes


def compute_iou(a: GroundTruth | Prediction, b: GroundTruth | Prediction) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    a_area = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    b_area = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = a_area + b_area - inter_area
    return inter_area / union if union > 0 else 0.0


def match_predictions(
    ground_truths: list[GroundTruth],
    predictions: list[Prediction],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    matches: list[dict[str, Any]] = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()

    ranked_predictions = sorted(enumerate(predictions), key=lambda item: item[1].confidence, reverse=True)
    for pred_idx, pred in ranked_predictions:
        best_gt_idx = None
        best_iou = 0.0
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in used_gt or gt.class_id != pred.class_id:
                continue
            iou = compute_iou(gt, pred)
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx is None:
            continue
        used_gt.add(best_gt_idx)
        used_pred.add(pred_idx)
        gt = ground_truths[best_gt_idx]
        matches.append({
            "gt_index": best_gt_idx,
            "pred_index": pred_idx,
            "class_id": pred.class_id,
            "confidence": pred.confidence,
            "iou": best_iou,
            "gt_box": gt,
            "pred_box": pred,
        })

    missed_gt = [idx for idx in range(len(ground_truths)) if idx not in used_gt]
    false_pred = [idx for idx in range(len(predictions)) if idx not in used_pred]
    return matches, missed_gt, false_pred


def resolve_dataset_inputs(
    data_yaml: Path | None,
    split: str,
    images_dir: Path | None,
    labels_dir: Path | None,
) -> tuple[Path, Path, dict[int, str]]:
    class_names = {}
    yaml_images_dir = None
    yaml_labels_dir = None
    
    if data_yaml:
        yaml_images_dir, yaml_labels_dir, class_names = load_data_yaml(data_yaml, split)
        
    final_images_dir = images_dir.resolve() if images_dir else yaml_images_dir
    
    if labels_dir:
        final_labels_dir = labels_dir.resolve()
    elif final_images_dir and final_images_dir.name == "images":
        final_labels_dir = final_images_dir.parent / "labels"
    else:
        final_labels_dir = yaml_labels_dir
        
    if not final_images_dir or not final_labels_dir:
        raise ValueError("Provide either valid --data or both --images-dir and --labels-dir")
        
    return final_images_dir, final_labels_dir, class_names


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run robust YOLO validation with CSV exports.")
    parser.add_argument("--config", default=None, help="Path to YAML config file.")
    parser.add_argument("--model", default=None, help="Path to model weights or exported model.")
    parser.add_argument("--data", default=None, help="Path to YOLO data.yaml.")
    parser.add_argument("--images-dir", default=None, help="Validation images directory.")
    parser.add_argument("--labels-dir", default=None, help="Validation labels directory.")
    parser.add_argument("--split", default="val", help="Dataset split to validate. Default: val")
    parser.add_argument("--output-dir", default=None, help="Directory for CSVs, JSON, and preview images.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching predictions to labels.")
    parser.add_argument("--device", default="cpu", help="Inference device.")
    parser.add_argument("--save-images", action="store_true", help="Save rendered prediction images.")
    return parser


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    if "robust_validation" in payload:
        section = payload["robust_validation"]
        if not isinstance(section, dict):
            raise ValueError(f"'robust_validation' must be a mapping in {config_path}")
        return section
    return payload


def _coalesce(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def resolve_args(cli_args: argparse.Namespace) -> argparse.Namespace:
    cfg: dict[str, Any] = {}
    if cli_args.config:
        cfg = _load_yaml_config(Path(cli_args.config).resolve())

    default_split = "val"
    default_imgsz = 640
    default_conf = 0.25
    default_iou = 0.5
    default_device = "cpu"

    merged = argparse.Namespace(
        model=_coalesce(cli_args.model, cfg.get("model")),
        data=_coalesce(cli_args.data, cfg.get("data")),
        images_dir=_coalesce(cli_args.images_dir, cfg.get("images_dir")),
        labels_dir=_coalesce(cli_args.labels_dir, cfg.get("labels_dir")),
        split=cfg.get("split", cli_args.split) if cli_args.split == default_split else cli_args.split,
        output_dir=_coalesce(cli_args.output_dir, cfg.get("output_dir")),
        imgsz=cfg.get("imgsz", cli_args.imgsz) if cli_args.imgsz == default_imgsz else cli_args.imgsz,
        conf=cfg.get("conf", cli_args.conf) if cli_args.conf == default_conf else cli_args.conf,
        iou=cfg.get("iou", cli_args.iou) if cli_args.iou == default_iou else cli_args.iou,
        device=cfg.get("device", cli_args.device) if cli_args.device == default_device else cli_args.device,
        save_images=bool(cli_args.save_images or cfg.get("save_images", False)),
        config=cli_args.config,
    )

    if not merged.model:
        raise ValueError("Missing model path. Provide --model or set 'model' in the config file.")
    if not merged.output_dir:
        raise ValueError("Missing output directory. Provide --output-dir or set 'output_dir' in the config file.")
    return merged


def _prediction_from_box(box: Any) -> Prediction:
    xyxy = box.xyxy[0].tolist()
    return Prediction(
        class_id=int(box.cls[0].item()),
        confidence=float(box.conf[0].item()),
        x1=float(xyxy[0]),
        y1=float(xyxy[1]),
        x2=float(xyxy[2]),
        y2=float(xyxy[3]),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    args = resolve_args(build_arg_parser().parse_args(argv))

    try:
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("robust validation requires pillow and ultralytics installed") from exc

    model_path = Path(args.model).resolve()
    data_yaml = Path(args.data).resolve() if args.data else None
    images_arg = Path(args.images_dir).resolve() if args.images_dir else None
    labels_arg = Path(args.labels_dir).resolve() if args.labels_dir else None
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir, labels_dir, class_names = resolve_dataset_inputs(data_yaml, args.split, images_arg, labels_arg)
    image_paths = collect_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No validation images found in {images_dir}")

    model = YOLO(str(model_path), task="detect")

    prediction_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    inference_times_ms: list[float] = []

    preview_dir = output_dir / "preview"
    if args.save_images:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        relative_path = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_path.with_suffix(".txt")

        with Image.open(image_path) as image:
            image_w, image_h = image.size

        ground_truths = load_ground_truth(label_path, image_w, image_h)
        total_gt += len(ground_truths)

        start = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
            save=args.save_images,
            project=str(output_dir),
            name="preview",
            exist_ok=True,
        )
        inference_ms = (time.perf_counter() - start) * 1000.0
        inference_times_ms.append(inference_ms)

        boxes = list(results[0].boxes) if results and getattr(results[0], "boxes", None) is not None else []
        predictions = [_prediction_from_box(box) for box in boxes]
        total_pred += len(predictions)

        matches, missed_gt, false_pred = match_predictions(ground_truths, predictions, args.iou)
        total_tp += len(matches)
        total_fp += len(false_pred)
        total_fn += len(missed_gt)

        matched_by_prediction = {item["pred_index"]: item for item in matches}
        for pred_index, pred in enumerate(predictions):
            matched = matched_by_prediction.get(pred_index)
            prediction_rows.append({
                "image_path": str(image_path),
                "label_path": str(label_path),
                "prediction_index": pred_index,
                "pred_class_id": pred.class_id,
                "pred_class_name": class_names.get(pred.class_id, str(pred.class_id)),
                "confidence": round(pred.confidence, 6),
                "x1": round(pred.x1, 3),
                "y1": round(pred.y1, 3),
                "x2": round(pred.x2, 3),
                "y2": round(pred.y2, 3),
                "matched": matched is not None,
                "matched_gt_index": matched["gt_index"] if matched else None,
                "match_iou": round(float(matched["iou"]), 6) if matched else None,
            })

        for match in matches:
            gt = match["gt_box"]
            pred = match["pred_box"]
            error_rows.append({
                "image_path": str(image_path),
                "error_type": "true_positive",
                "class_id": match["class_id"],
                "class_name": class_names.get(match["class_id"], str(match["class_id"])),
                "gt_index": match["gt_index"],
                "pred_index": match["pred_index"],
                "confidence": round(float(match["confidence"]), 6),
                "iou": round(float(match["iou"]), 6),
                "gt_x1": round(gt.x1, 3),
                "gt_y1": round(gt.y1, 3),
                "gt_x2": round(gt.x2, 3),
                "gt_y2": round(gt.y2, 3),
                "pred_x1": round(pred.x1, 3),
                "pred_y1": round(pred.y1, 3),
                "pred_x2": round(pred.x2, 3),
                "pred_y2": round(pred.y2, 3),
            })

        for gt_index in missed_gt:
            gt = ground_truths[gt_index]
            error_rows.append({
                "image_path": str(image_path),
                "error_type": "false_negative",
                "class_id": gt.class_id,
                "class_name": class_names.get(gt.class_id, str(gt.class_id)),
                "gt_index": gt_index,
                "pred_index": None,
                "confidence": None,
                "iou": None,
                "gt_x1": round(gt.x1, 3),
                "gt_y1": round(gt.y1, 3),
                "gt_x2": round(gt.x2, 3),
                "gt_y2": round(gt.y2, 3),
                "pred_x1": None,
                "pred_y1": None,
                "pred_x2": None,
                "pred_y2": None,
            })

        for pred_index in false_pred:
            pred = predictions[pred_index]
            error_rows.append({
                "image_path": str(image_path),
                "error_type": "false_positive",
                "class_id": pred.class_id,
                "class_name": class_names.get(pred.class_id, str(pred.class_id)),
                "gt_index": None,
                "pred_index": pred_index,
                "confidence": round(pred.confidence, 6),
                "iou": None,
                "gt_x1": None,
                "gt_y1": None,
                "gt_x2": None,
                "gt_y2": None,
                "pred_x1": round(pred.x1, 3),
                "pred_y1": round(pred.y1, 3),
                "pred_x2": round(pred.x2, 3),
                "pred_y2": round(pred.y2, 3),
            })

        image_rows.append({
            "image_path": str(image_path),
            "label_path": str(label_path),
            "ground_truth_count": len(ground_truths),
            "prediction_count": len(predictions),
            "true_positive_count": len(matches),
            "false_positive_count": len(false_pred),
            "false_negative_count": len(missed_gt),
            "image_has_error": bool(false_pred or missed_gt),
            "inference_time_ms": round(inference_ms, 3),
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_inference_ms = sum(inference_times_ms) / len(inference_times_ms)

    # ---- Per-class metrics ----
    class_tp: dict[int, int] = defaultdict(int)
    class_fp: dict[int, int] = defaultdict(int)
    class_fn: dict[int, int] = defaultdict(int)
    for row in error_rows:
        cid = row["class_id"]
        if row["error_type"] == "true_positive":
            class_tp[cid] += 1
        elif row["error_type"] == "false_positive":
            class_fp[cid] += 1
        elif row["error_type"] == "false_negative":
            class_fn[cid] += 1

    # Include all classes from class_names as well to ensure zero-count classes are reported
    all_class_ids = sorted(set(class_tp) | set(class_fp) | set(class_fn) | set(class_names.keys()))
    per_class_rows: list[dict[str, Any]] = []
    for cid in all_class_ids:
        tp_c = class_tp[cid]
        fp_c = class_fp[cid]
        fn_c = class_fn[cid]
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
        f1_c = (2 * p_c * r_c / (p_c + r_c)) if (p_c + r_c) else 0.0
        per_class_rows.append({
            "class_id": cid,
            "class_name": class_names.get(cid, str(cid)),
            "true_positives": tp_c,
            "false_positives": fp_c,
            "false_negatives": fn_c,
            "precision": round(p_c, 6),
            "recall": round(r_c, 6),
            "f1": round(f1_c, 6),
        })

    # ---- Confusion matrix ----
    # Rows = GT class, Cols = Pred class (only for matched pairs)
    confusion: dict[tuple[int, int], int] = defaultdict(int)
    for row in error_rows:
        if row["error_type"] == "true_positive":
            confusion[(row["class_id"], row["class_id"])] += 1
        elif row["error_type"] == "false_positive" and row["class_id"] is not None:
            confusion[(-1, row["class_id"])] += 1  # -1 = background
        elif row["error_type"] == "false_negative" and row["class_id"] is not None:
            confusion[(row["class_id"], -1)] += 1  # -1 = background

    cm_ids = sorted(set(all_class_ids) | {-1})
    cm_rows: list[dict[str, Any]] = []
    for gt_id in cm_ids:
        row_dict: dict[str, Any] = {
            "gt_class_id": gt_id,
            "gt_class_name": "background" if gt_id == -1 else class_names.get(gt_id, str(gt_id)),
        }
        for pred_id in cm_ids:
            col_name = f"pred_{pred_id}" if pred_id != -1 else "pred_background"
            row_dict[col_name] = confusion.get((gt_id, pred_id), 0)
        cm_rows.append(row_dict)

    summary = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path),
        "data_yaml": str(data_yaml) if data_yaml else None,
        "split": args.split,
        "images_dir": str(images_dir),
        "labels_dir": str(labels_dir),
        "image_count": len(image_paths),
        "ground_truth_count": total_gt,
        "prediction_count": total_pred,
        "true_positive_count": total_tp,
        "false_positive_count": total_fp,
        "false_negative_count": total_fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_inference_time_ms": round(mean_inference_ms, 3),
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "per_class_metrics": per_class_rows,
        "preview_dir": str(preview_dir) if args.save_images else None,
    }

    # ---- Write all outputs ----
    _write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        [
            "image_path", "label_path", "prediction_index",
            "pred_class_id", "pred_class_name", "confidence",
            "x1", "y1", "x2", "y2",
            "matched", "matched_gt_index", "match_iou",
        ],
    )
    _write_csv(
        output_dir / "errors.csv",
        error_rows,
        [
            "image_path", "error_type", "class_id", "class_name",
            "gt_index", "pred_index", "confidence", "iou",
            "gt_x1", "gt_y1", "gt_x2", "gt_y2",
            "pred_x1", "pred_y1", "pred_x2", "pred_y2",
        ],
    )
    _write_csv(
        output_dir / "per_image_summary.csv",
        image_rows,
        [
            "image_path", "label_path",
            "ground_truth_count", "prediction_count",
            "true_positive_count", "false_positive_count", "false_negative_count",
            "image_has_error", "inference_time_ms",
        ],
    )
    _write_csv(
        output_dir / "per_class_metrics.csv",
        per_class_rows,
        ["class_id", "class_name", "true_positives", "false_positives",
         "false_negatives", "precision", "recall", "f1"],
    )
    if cm_rows:
        _write_csv(output_dir / "confusion_matrix.csv", cm_rows, list(cm_rows[0].keys()))

    # Summary JSON
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Summary CSV (single-row for thesis tables)
    summary_csv_row = {
        "model": model_path.name,
        "image_count": len(image_paths),
        "gt_count": total_gt,
        "pred_count": total_pred,
        "TP": total_tp, "FP": total_fp, "FN": total_fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_inference_ms": round(mean_inference_ms, 3),
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
    }
    _write_csv(output_dir / "summary.csv", [summary_csv_row], list(summary_csv_row.keys()))

    print(f"\nValidation complete.")
    print(f"  Images     : {len(image_paths)}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1         : {f1:.4f}")
    print(f"  Results in : {output_dir}")


if __name__ == "__main__":
    main()
