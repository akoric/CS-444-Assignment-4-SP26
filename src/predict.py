import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from .constants import VOC_CLASSES, IMAGENET_MEAN, IMAGENET_STD, YOLO_IMG_DIM


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Detection:
    image_id: str
    class_name: str
    score: float
    box: np.ndarray  # [x1, y1, x2, y2]


# -----------------------------
# Preprocessing
# -----------------------------
def load_image(image_name: str, root_img_directory: str) -> Tuple[np.ndarray, str]:
    """
    Load image from disk and return (image, resolved_path).
    """
    image_path = (
        image_name
        if os.path.isabs(image_name)
        else os.path.join(root_img_directory, image_name)
    )

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    return image, image_path


def preprocess_image(image: np.ndarray, input_size: int = YOLO_IMG_DIM) -> torch.Tensor:
    """
    Resize, normalize, and convert to tensor.
    """
    image = cv2.resize(image, (input_size, input_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Normalize
    image /= 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float()

    return image


# -----------------------------
# Decoding YOLO output
# -----------------------------
def decode_yolo_output(
    pred: torch.Tensor,
    conf_threshold: float = 0.1,
    B: int = 2,
) -> Tuple[List[torch.Tensor], List[int], List[float]]:
    """
    Decode YOLOv1-style output.

    Assumptions:
    - S x S x (B*5 + C) layout
    - B = 2
    - First 10 values: 2 boxes (5 each)
    - Remaining: class probabilities
    """
    pred = pred.squeeze(0)  # (S, S, 30)
    S = pred.shape[0]

    boxes = []
    cls_indices = []
    scores = []

    cell_size = 1.0 / S

    for i in range(S):
        for j in range(S):
            cell = pred[i, j]

            for b in range(B):
                offset = b * 5
                conf = cell[offset + 4]

                if conf.item() < conf_threshold:
                    continue

                cx = (cell[offset + 0] + j) * cell_size
                cy = (cell[offset + 1] + i) * cell_size
                w = cell[offset + 2]
                h = cell[offset + 3]

                class_probs = cell[B * 5 :]
                class_score, class_idx = torch.max(class_probs, dim=0)

                score = (conf * class_score).item()

                if score < conf_threshold:
                    continue

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                box = torch.stack((x1, y1, x2, y2)).clamp_(0.0, 1.0)
                boxes.append(box)
                cls_indices.append(int(class_idx.item()))
                scores.append(score)

    return boxes, cls_indices, scores


def apply_nms(
    boxes: List[torch.Tensor],
    scores: List[float],
    cls_indices: List[int],
    iou_threshold: float = 0.5,
) -> Tuple[List[torch.Tensor], List[int], List[float]]:
    """
    Apply class-wise NMS using torchvision.
    """
    if len(boxes) == 0:
        return [], [], []

    boxes = torch.stack(boxes)
    scores = boxes.new_tensor(scores)

    keep_all = []

    for cls in set(cls_indices):
        inds = [i for i, c in enumerate(cls_indices) if c == cls]
        cls_boxes = boxes[inds]
        cls_scores = scores[inds]

        keep = nms(cls_boxes, cls_scores, iou_threshold)
        keep_all.extend([inds[int(i)] for i in keep.tolist()])

    keep_all = sorted(keep_all, key=lambda idx: scores[idx].item(), reverse=True)

    return (
        [boxes[i] for i in keep_all],
        [cls_indices[i] for i in keep_all],
        [scores[i].item() for i in keep_all],
    )


# -----------------------------
# Main prediction function
# -----------------------------
@torch.inference_mode()
def predict_image(
    model,
    image_name: str,
    root_img_directory: str = "",
    conf_threshold: float = 0.1,
    nms_iou: float = 0.5,
) -> List[Detection]:
    """
    Run inference on a single image and return structured detections.
    """
    image, image_path = load_image(image_name, root_img_directory)
    h, w, _ = image.shape

    img_tensor = preprocess_image(image)
    img_tensor = img_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    pred = model(img_tensor)
    if isinstance(pred, dict):
        raise TypeError(
            "predict_image currently expects a YOLO-style tensor output, not a DETR-style dict"
        )

    boxes, cls_indices, scores = decode_yolo_output(pred, conf_threshold)
    boxes, cls_indices, scores = apply_nms(boxes, scores, cls_indices, nms_iou)

    detections: List[Detection] = []

    for box, cls_idx, score in zip(boxes, cls_indices, scores):
        x1, y1, x2, y2 = box

        # Rescale to original image size
        x1 = int(x1.clamp_(0.0, 1.0).item() * w)
        x2 = int(x2.clamp_(0.0, 1.0).item() * w)
        y1 = int(y1.clamp_(0.0, 1.0).item() * h)
        y2 = int(y2.clamp_(0.0, 1.0).item() * h)

        detections.append(
            Detection(
                image_id=image_name,
                class_name=VOC_CLASSES[cls_idx],
                score=float(score),
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
            )
        )

    return detections


@torch.inference_mode()
def predict_image_detr(
    model,
    image_name: str,
    root_img_directory: str = "",
    conf_threshold: float = 0.05,
    max_size: int = 384,
) -> List[Detection]:
    """
    Run DETR inference on a single image and return structured detections.
    """
    image, _ = load_image(image_name, root_img_directory)
    h_orig, w_orig = image.shape[:2]

    scale = float(max_size) / float(max(h_orig, w_orig))
    if scale < 1.0:
        new_h = max(1, int(round(h_orig * scale)))
        new_w = max(1, int(round(w_orig * scale)))
        image_resized = cv2.resize(image, (new_w, new_h))
    else:
        image_resized = image

    image_rgb = (
        cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    )
    image_norm = (image_rgb - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32
    )
    img_tensor = torch.from_numpy(np.transpose(image_norm, (2, 0, 1))).float()

    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)

    outputs = model([img_tensor])
    pred_logits = outputs["pred_logits"][0]  # [Q, C+1]
    pred_boxes = outputs["pred_boxes"][0]  # [Q, 4] normalized cx,cy,w,h

    probs = pred_logits.softmax(-1)
    scores, cls_indices = probs[:, :-1].max(-1)  # exclude background class

    detections: List[Detection] = []
    for i in range(scores.shape[0]):
        score = float(scores[i].item())
        if score < conf_threshold:
            continue
        cls_idx = int(cls_indices[i].item())
        cx, cy, w, h = pred_boxes[i].tolist()

        x1 = int(max(0.0, cx - w / 2) * w_orig)
        y1 = int(max(0.0, cy - h / 2) * h_orig)
        x2 = int(min(1.0, cx + w / 2) * w_orig)
        y2 = int(min(1.0, cy + h / 2) * h_orig)

        detections.append(
            Detection(
                image_id=image_name,
                class_name=VOC_CLASSES[cls_idx],
                score=score,
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
            )
        )

    return detections
