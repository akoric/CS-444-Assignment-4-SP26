from pathlib import Path

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

from .constants import VOC_CLASSES, IMAGENET_MEAN, IMAGENET_STD


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 4))
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0), dim=-1)


def resize_image_and_boxes(
    image: torch.Tensor,
    boxes_xyxy_abs: torch.Tensor,
    image_size: int,
    keep_aspect_ratio: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, h, w = image.shape

    if keep_aspect_ratio:
        scale = float(image_size) / float(max(h, w))
        if scale >= 1.0:
            return image, boxes_xyxy_abs

        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        image = F.resize(image, size=[new_h, new_w])

        if boxes_xyxy_abs.numel() > 0:
            boxes_xyxy_abs = boxes_xyxy_abs * boxes_xyxy_abs.new_tensor(
                [scale, scale, scale, scale]
            )
        return image, boxes_xyxy_abs

    if h == image_size and w == image_size:
        return image, boxes_xyxy_abs

    image = F.resize(image, size=[image_size, image_size])
    if boxes_xyxy_abs.numel() > 0:
        boxes_xyxy_abs = boxes_xyxy_abs * boxes_xyxy_abs.new_tensor(
            [image_size / w, image_size / h, image_size / w, image_size / h]
        )
    return image, boxes_xyxy_abs


def clip_and_filter_boxes(
    boxes_xyxy_abs: torch.Tensor,
    labels: torch.Tensor,
    height: int,
    width: int,
    min_size: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes_xyxy_abs.numel() == 0:
        return boxes_xyxy_abs.reshape(0, 4), labels[:0]

    boxes = boxes_xyxy_abs.clone()
    boxes[:, 0::2] = boxes[:, 0::2].clamp_(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp_(0, height)

    wh = boxes[:, 2:] - boxes[:, :2]
    keep = (wh[:, 0] >= min_size) & (wh[:, 1] >= min_size)

    return boxes[keep], labels[keep]


class DetectionAugment:
    """
    torchvision v2 augmentation pipeline for detection.
    Applies the same transform jointly to image + boxes.
    """

    def __init__(self):
        fill_rgb = tuple(int(round(c * 255)) for c in IMAGENET_MEAN)
        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomApply(
                    [
                        v2.RandomAffine(
                            degrees=0,
                            translate=(0.2, 0.2),
                            scale=(0.8, 1.2),
                            fill=fill_rgb,
                        )
                    ],
                    p=0.5,
                ),
                v2.RandomApply(
                    [v2.RandomIoUCrop(min_scale=0.6, trials=20)],
                    p=0.5,
                ),
                v2.SanitizeBoundingBoxes(),
            ]
        )

    def __call__(self, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor):
        """
        image: uint8 tensor, shape [3, H, W]
        boxes: float tensor, absolute xyxy, shape [N, 4]
        labels: long tensor, shape [N]
        """
        if boxes.numel() == 0:
            return image, boxes.reshape(0, 4), labels[:0]

        h, w = image.shape[-2:]
        target = {
            "boxes": tv_tensors.BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=(h, w),
            ),
            "labels": labels,
        }

        image, target = self.transform(image, target)
        return image, target["boxes"].as_subclass(torch.Tensor), target["labels"]


class VOCDetectionDataset(data.Dataset):
    def __init__(
        self,
        root_img_dir: Path,
        dataset_file: Path,
        train: bool,
        detector_type: str = "detr",  # "detr" or "yolo"
        backbone: str = "resnet18",
        image_size: int = 448,
        augmentation: bool = True,
        grid_size: int = 14,
        keep_aspect_ratio: bool | None = None,
    ) -> None:
        self.root = Path(root_img_dir)
        self.dataset_file = Path(dataset_file)
        self.train = bool(train)
        self.detector_type = detector_type.lower()
        self.backbone = str(backbone)
        self.image_size = int(image_size)
        self.augmentation = bool(augmentation)
        self.grid_size = int(grid_size)

        if self.detector_type not in {"detr", "yolo"}:
            raise ValueError(f"Unsupported detector_type={self.detector_type}")

        if keep_aspect_ratio is None:
            self.keep_aspect_ratio = (
                self.detector_type == "detr" and "resnet" not in self.backbone.lower()
            )
        else:
            self.keep_aspect_ratio = bool(keep_aspect_ratio)

        self.augment = DetectionAugment()

        self.fnames: list[str] = []
        self.boxes: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []

        with self.dataset_file.open() as f:
            for line in f:
                split_line = line.strip().split()
                if not split_line:
                    continue

                self.fnames.append(split_line[0])
                num_boxes = (len(split_line) - 1) // 5

                boxes = []
                labels = []
                for i in range(num_boxes):
                    x1 = float(split_line[1 + 5 * i])
                    y1 = float(split_line[2 + 5 * i])
                    x2 = float(split_line[3 + 5 * i])
                    y2 = float(split_line[4 + 5 * i])
                    cls = int(split_line[5 + 5 * i])
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

                self.boxes.append(torch.tensor(boxes, dtype=torch.float32))
                self.labels.append(torch.tensor(labels, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.fnames)

    def __getitem__(self, idx: int):
        fname = self.fnames[idx]
        img_path = self.root / fname

        # cv2 loads BGR HWC uint8
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(
            2, 0, 1
        )  # [3,H,W], uint8
        _, orig_h, orig_w = image.shape

        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()

        if self.train and self.augmentation:
            image, boxes, labels = self.augment(image, boxes, labels)

        _, h, w = image.shape
        boxes, labels = clip_and_filter_boxes(boxes, labels, h, w)

        image, boxes = resize_image_and_boxes(
            image=image,
            boxes_xyxy_abs=boxes,
            image_size=self.image_size,
            keep_aspect_ratio=self.keep_aspect_ratio,
        )

        _, h, w = image.shape
        boxes, labels = clip_and_filter_boxes(boxes, labels, h, w)

        if boxes.numel() > 0:
            scale = boxes.new_tensor([w, h, w, h])
            boxes_xyxy_norm = (boxes / scale).clamp_(0.0, 1.0)
        else:
            scale = torch.tensor([w, h, w, h], dtype=torch.float32)
            boxes_xyxy_norm = torch.zeros((0, 4), dtype=torch.float32)

        if self.detector_type == "detr":
            image = image.float() / 255.0
            image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            target = self.format_target_for_detr(
                fname=fname,
                labels=labels,
                boxes_xyxy_norm=boxes_xyxy_norm,
                orig_height=orig_h,
                orig_width=orig_w,
                height=h,
                width=w,
            )
            return image, target

        # YOLO path
        image = image.float() / 255.0
        image = F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        target = self.format_target_for_yolo(
            boxes_xyxy_norm=boxes_xyxy_norm,
            labels=labels,
        )
        return image, target

    def format_target_for_detr(
        self,
        fname: str,
        labels: torch.Tensor,
        boxes_xyxy_norm: torch.Tensor,
        orig_height: int,
        orig_width: int,
        height: int,
        width: int,
    ):
        boxes_cxcywh = xyxy_to_cxcywh(boxes_xyxy_norm).clamp_(0.0, 1.0)
        orig_size_hw = torch.tensor([orig_height, orig_width], dtype=torch.int64)
        size_hw = torch.tensor([height, width], dtype=torch.int64)

        return {
            "image_id": fname,
            "labels": labels,  # 0..19
            "boxes": boxes_cxcywh,  # normalized cxcywh
            "orig_size": orig_size_hw,
            "size": size_hw.clone(),
            "boxes_xyxy_abs": boxes_xyxy_norm
            * boxes_xyxy_norm.new_tensor([width, height, width, height]),
        }

    def format_target_for_yolo(
        self,
        boxes_xyxy_norm: torch.Tensor,
        labels: torch.Tensor,
    ):
        S = self.grid_size
        C = len(VOC_CLASSES)

        target = torch.zeros((S, S, 5 + C), dtype=torch.float32)
        cell_size = 1.0 / S

        if boxes_xyxy_norm.numel() == 0:
            return {
                "target_boxes": target[:, :, :4],
                "target_cls": target[:, :, 5:],
                "has_object_map": target[:, :, 4] > 0,
            }

        wh = boxes_xyxy_norm[:, 2:] - boxes_xyxy_norm[:, :2]
        centers = (boxes_xyxy_norm[:, 2:] + boxes_xyxy_norm[:, :2]) / 2

        for i in range(centers.size(0)):
            center_xy = centers[i]
            ij = (center_xy / cell_size).ceil() - 1
            ij = ij.clamp(min=0, max=S - 1)

            row = int(ij[1])
            col = int(ij[0])

            xy0 = ij * cell_size
            delta_xy = (center_xy - xy0) / cell_size

            target[row, col, 0:2] = delta_xy
            target[row, col, 2:4] = wh[i]
            target[row, col, 4] = 1.0
            target[row, col, 5 + int(labels[i])] = 1.0

        return {
            "target_boxes": target[:, :, :4],
            "target_cls": target[:, :, 5:],
            "has_object_map": target[:, :, 4] > 0,
        }


def collate_fn_detr(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def collate_fn_yolo(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, {
        "target_boxes": torch.stack([t["target_boxes"] for t in targets], dim=0),
        "target_cls": torch.stack([t["target_cls"] for t in targets], dim=0),
        "has_object_map": torch.stack([t["has_object_map"] for t in targets], dim=0),
    }
