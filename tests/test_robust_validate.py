from __future__ import annotations

from pathlib import Path

from yoloml.validations.robust_validate import (
    GroundTruth,
    Prediction,
    collect_images,
    compute_iou,
    load_data_yaml,
    load_ground_truth,
    match_predictions,
    resolve_dataset_inputs,
    resolve_args,
)


def test_load_data_yaml_and_collect_images(yolo_dataset: Path):
    images_dir, labels_dir, class_names = load_data_yaml(yolo_dataset / "data.yaml", "val")

    assert images_dir == (yolo_dataset / "images" / "val").resolve()
    assert labels_dir == (yolo_dataset / "labels" / "val").resolve()
    assert class_names == {0: "leaf"}
    assert collect_images(images_dir) == [(yolo_dataset / "images" / "val" / "val_0.jpg").resolve()]


def test_load_ground_truth_converts_normalized_labels(yolo_dataset: Path):
    label_path = yolo_dataset / "labels" / "val" / "val_0.txt"

    boxes = load_ground_truth(label_path, image_w=100, image_h=200)

    assert len(boxes) == 1
    box = boxes[0]
    assert box.class_id == 0
    assert box.x1 == 25.0
    assert box.y1 == 50.0
    assert box.x2 == 75.0
    assert box.y2 == 150.0


def test_compute_iou_and_matching():
    ground_truths = [GroundTruth(class_id=0, x1=10, y1=10, x2=30, y2=30)]
    predictions = [
        Prediction(class_id=0, confidence=0.95, x1=11, y1=11, x2=29, y2=29),
        Prediction(class_id=0, confidence=0.40, x1=60, y1=60, x2=90, y2=90),
    ]

    iou = compute_iou(ground_truths[0], predictions[0])
    matches, missed_gt, false_pred = match_predictions(ground_truths, predictions, iou_threshold=0.5)

    assert iou > 0.8
    assert len(matches) == 1
    assert missed_gt == []
    assert false_pred == [1]


def test_match_predictions_respects_class_id():
    ground_truths = [GroundTruth(class_id=1, x1=10, y1=10, x2=30, y2=30)]
    predictions = [Prediction(class_id=0, confidence=0.99, x1=10, y1=10, x2=30, y2=30)]

    matches, missed_gt, false_pred = match_predictions(ground_truths, predictions, iou_threshold=0.5)

    assert matches == []
    assert missed_gt == [0]
    assert false_pred == [0]


def test_resolve_dataset_inputs_without_data_yaml(yolo_dataset: Path):
    images_dir, labels_dir, class_names = resolve_dataset_inputs(
        data_yaml=None,
        split="val",
        images_dir=yolo_dataset / "images" / "val",
        labels_dir=yolo_dataset / "labels" / "val",
    )

    assert images_dir == (yolo_dataset / "images" / "val").resolve()
    assert labels_dir == (yolo_dataset / "labels" / "val").resolve()
    assert class_names == {}


def test_resolve_args_reads_yaml_config(tmp_path: Path):
    config_path = tmp_path / "robust_validation.yaml"
    config_path.write_text(
        (
            "robust_validation:\n"
            "  model: C:/models/best.pt\n"
            "  images_dir: C:/data/images/val\n"
            "  labels_dir: C:/data/labels/val\n"
            "  output_dir: C:/outputs/robust_validation\n"
            "  imgsz: 512\n"
            "  conf: 0.3\n"
            "  iou: 0.55\n"
            "  device: cpu\n"
            "  save_images: true\n"
        ),
        encoding="utf-8",
    )

    parser = {
        "config": str(config_path),
        "model": None,
        "data": None,
        "images_dir": None,
        "labels_dir": None,
        "split": "val",
        "output_dir": None,
        "imgsz": 640,
        "conf": 0.25,
        "iou": 0.5,
        "device": "cpu",
        "save_images": False,
    }
    args = resolve_args(type("Args", (), parser)())

    assert args.model == "C:/models/best.pt"
    assert args.images_dir == "C:/data/images/val"
    assert args.labels_dir == "C:/data/labels/val"
    assert args.output_dir == "C:/outputs/robust_validation"
    assert args.imgsz == 512
    assert args.conf == 0.3
    assert args.iou == 0.55
    assert args.save_images is True
