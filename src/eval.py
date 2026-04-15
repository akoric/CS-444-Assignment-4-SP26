import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, List, Sequence, Tuple

import numpy as np

from .constants import VOC_CLASSES
from .predict import predict_image


@dataclass(frozen=True)
class Prediction:
    image_id: str
    confidence: float
    box: np.ndarray  # [x1, y1, x2, y2]


def voc_ap(rec: np.ndarray, prec: np.ndarray, use_07_metric: bool = False) -> float:
    """
    Compute VOC Average Precision.
    If use_07_metric=True, uses the VOC 2007 11-point interpolation method.
    """
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0.0
            else:
                p = float(np.max(prec[rec >= t]))
            ap += p / 11.0
        return ap

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    changing_points = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(
        np.sum(
            (mrec[changing_points + 1] - mrec[changing_points])
            * mpre[changing_points + 1]
        )
    )
    return ap


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two boxes in VOC format [x1, y1, x2, y2].
    """
    ixmin = max(box_a[0], box_b[0])
    iymin = max(box_a[1], box_b[1])
    ixmax = min(box_a[2], box_b[2])
    iymax = min(box_a[3], box_b[3])

    iw = max(ixmax - ixmin + 1.0, 0.0)
    ih = max(iymax - iymin + 1.0, 0.0)
    inter = iw * ih

    area_a = (box_a[2] - box_a[0] + 1.0) * (box_a[3] - box_a[1] + 1.0)
    area_b = (box_b[2] - box_b[0] + 1.0) * (box_b[3] - box_b[1] + 1.0)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0

    return float(inter / union)


def parse_test_dataset_file(
    test_dataset_file: str,
) -> Tuple[List[str], DefaultDict[Tuple[str, str], List[np.ndarray]]]:
    """
    Parse dataset text file.

    Expected row format:
        image_id x1 y1 x2 y2 cls x1 y1 x2 y2 cls ...

    Returns:
        image_ids: list of image ids / paths
        targets: {(image_id, class_name): [box, ...]}
    """
    targets: DefaultDict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
    image_ids: List[str] = []

    with open(test_dataset_file, "r") as f:
        lines = [line.strip().split() for line in f if line.strip()]

    for row in lines:
        image_id = row[0]
        image_ids.append(image_id)

        num_objects = (len(row) - 1) // 5
        for i in range(num_objects):
            x1 = int(row[1 + 5 * i])
            y1 = int(row[2 + 5 * i])
            x2 = int(row[3 + 5 * i])
            y2 = int(row[4 + 5 * i])
            class_idx = int(row[5 + 5 * i])

            class_name = VOC_CLASSES[class_idx]
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            targets[(image_id, class_name)].append(box)

    return image_ids, targets


def collect_predictions(
    model,
    image_ids: Sequence[str],
    img_root: str,
    conf_threshold: float = 0.1,
    nms_iou: float = 0.5,
) -> DefaultDict[str, List[Prediction]]:
    """
    Run inference over all images and collect predictions grouped by class.
    """
    preds: DefaultDict[str, List[Prediction]] = defaultdict(list)

    model.eval()
    for image_path in image_ids:
        results = predict_image(
            model,
            image_path,
            root_img_directory=img_root,
            conf_threshold=conf_threshold,
            nms_iou=nms_iou,
        )

        for det in results:
            preds[det.class_name].append(
                Prediction(
                    image_id=det.image_id,
                    confidence=det.score,
                    box=det.box,
                )
            )

    return preds


def evaluate_class(
    predictions: Sequence[Prediction],
    targets: Dict[Tuple[str, str], List[np.ndarray]],
    class_name: str,
    iou_threshold: float = 0.5,
    use_07_metric: bool = False,
) -> Dict[str, float]:
    """
    Evaluate one class and return AP/precision/recall statistics.
    """
    gt_for_class = {
        image_id: boxes
        for (image_id, cls), boxes in targets.items()
        if cls == class_name
    }

    npos = sum(len(boxes) for boxes in gt_for_class.values())

    if len(predictions) == 0:
        return {
            "ap": 0.0,
            "final_precision": 0.0,
            "final_recall": 0.0,
            "num_predictions": 0,
            "num_gt": int(npos),
        }

    predictions = sorted(predictions, key=lambda p: p.confidence, reverse=True)

    # Track which GT boxes have already been matched.
    matched = {
        image_id: np.zeros(len(boxes), dtype=bool)
        for image_id, boxes in gt_for_class.items()
    }

    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)

    for i, pred in enumerate(predictions):
        gt_boxes = gt_for_class.get(pred.image_id)

        if gt_boxes is None or len(gt_boxes) == 0:
            fp[i] = 1.0
            continue

        ious = np.array(
            [compute_iou(pred.box, gt_box) for gt_box in gt_boxes], dtype=np.float32
        )
        best_gt_idx = int(np.argmax(ious))
        best_iou = float(ious[best_gt_idx])

        if best_iou >= iou_threshold and not matched[pred.image_id][best_gt_idx]:
            tp[i] = 1.0
            matched[pred.image_id][best_gt_idx] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    if npos > 0:
        rec = tp_cum / float(npos)
    else:
        rec = np.zeros_like(tp_cum)

    prec = tp_cum / np.maximum(tp_cum + fp_cum, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric=use_07_metric)

    return {
        "ap": float(ap),
        "final_precision": float(prec[-1]) if len(prec) > 0 else 0.0,
        "final_recall": float(rec[-1]) if len(rec) > 0 else 0.0,
        "num_predictions": int(len(predictions)),
        "num_gt": int(npos),
    }


def format_results_table(
    class_results: Dict[str, Dict[str, float]], map_value: float
) -> str:
    """
    Build a nicely formatted text table for printing.
    """
    header = (
        "\nVOC Evaluation Results\n"
        + "=" * 79
        + "\n"
        + f"{'Class':<18} {'AP':>10} {'Prec':>10} {'Recall':>10} {'#Pred':>10} {'#GT':>10}\n"
        + "-" * 79
    )

    rows = []
    for class_name, metrics in class_results.items():
        rows.append(
            f"{class_name:<18} "
            f"{metrics['ap']:>10.4f} "
            f"{metrics['final_precision']:>10.4f} "
            f"{metrics['final_recall']:>10.4f} "
            f"{metrics['num_predictions']:>10d} "
            f"{metrics['num_gt']:>10d}"
        )

    footer = "-" * 79 + f"\n{'mAP':<18} {map_value:>10.4f}\n" + "=" * 79

    return "\n".join([header] + rows + [footer])


def voc_eval(
    preds: Dict[str, List[Prediction]],
    targets: Dict[Tuple[str, str], List[np.ndarray]],
    voc_classes: Sequence[str] = VOC_CLASSES,
    iou_threshold: float = 0.5,
    use_07_metric: bool = False,
    print_results: bool = True,
) -> Dict[str, object]:
    """
    Evaluate predictions across all VOC classes.

    Returns:
        {
            "aps": [...],
            "map": float,
            "per_class": {
                class_name: {
                    "ap": ...,
                    "final_precision": ...,
                    "final_recall": ...,
                    "num_predictions": ...,
                    "num_gt": ...,
                }
            }
        }
    """
    per_class_results: Dict[str, Dict[str, float]] = {}
    aps: List[float] = []

    for class_name in voc_classes:
        class_metrics = evaluate_class(
            predictions=preds.get(class_name, []),
            targets=targets,
            class_name=class_name,
            iou_threshold=iou_threshold,
            use_07_metric=use_07_metric,
        )
        per_class_results[class_name] = class_metrics
        aps.append(class_metrics["ap"])

    map_value = float(np.mean(aps)) if aps else 0.0

    if print_results:
        print(format_results_table(per_class_results, map_value))
        sys.stdout.flush()

    return {
        "aps": aps,
        "map": map_value,
        "per_class": per_class_results,
    }


def evaluate(
    model,
    test_dataset_file: str,
    img_root: str,
    conf_threshold: float = 0.1,
    nms_iou: float = 0.5,
    iou_threshold: float = 0.5,
    use_07_metric: bool = False,
    print_results: bool = True,
) -> Dict[str, object]:
    """
    End-to-end evaluation on a VOC-style dataset file.
    """
    image_ids, targets = parse_test_dataset_file(test_dataset_file)

    if print_results:
        print(f"Evaluating on {len(image_ids)} images...")
        sys.stdout.flush()

    preds = collect_predictions(
        model=model,
        image_ids=image_ids,
        img_root=img_root,
        conf_threshold=conf_threshold,
        nms_iou=nms_iou,
    )

    results = voc_eval(
        preds=preds,
        targets=targets,
        voc_classes=VOC_CLASSES,
        iou_threshold=iou_threshold,
        use_07_metric=use_07_metric,
        print_results=print_results,
    )

    return results


def evaluate_detr(
    model,
    test_dataset_file: str,
    img_root: str,
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    use_07_metric: bool = False,
    print_results: bool = True,
    max_size: int = 384,
) -> Dict[str, object]:
    """
    End-to-end DETR evaluation on a VOC-style dataset file.
    """
    from .predict import predict_image_detr

    image_ids, targets = parse_test_dataset_file(test_dataset_file)

    if print_results:
        print(f"Evaluating on {len(image_ids)} images...")
        sys.stdout.flush()

    preds: DefaultDict[str, List[Prediction]] = defaultdict(list)
    model.eval()
    for image_path in image_ids:
        results = predict_image_detr(
            model=model,
            image_name=image_path,
            root_img_directory=img_root,
            conf_threshold=conf_threshold,
            max_size=max_size,
        )
        for det in results:
            preds[det.class_name].append(
                Prediction(
                    image_id=det.image_id,
                    confidence=det.score,
                    box=det.box,
                )
            )

    return voc_eval(
        preds=preds,
        targets=targets,
        voc_classes=VOC_CLASSES,
        iou_threshold=iou_threshold,
        use_07_metric=use_07_metric,
        print_results=print_results,
    )
