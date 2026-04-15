from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(x):
    """Convert boxes from center-size form to corner form.

    Args:
        x: Tensor of shape [..., 4] containing boxes as [cx, cy, w, h].

    Returns:
        Tensor with the same shape containing [x1, y1, x2, y2] boxes.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def _box_area_xyxy(boxes):
    """Compute box areas for boxes stored as [x1, y1, x2, y2]."""
    # clamp prevents negative area if a box becomes malformed
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(
        min=0
    )


def compute_total_loss(loss_dict, weight_dict):
    """Combine named DETR loss terms using the provided scalar weights.

    Args:
        loss_dict: Dictionary such as {"loss_ce": ..., "loss_bbox": ...}.
        weight_dict: Dictionary mapping loss names to scalar coefficients.

    Returns:
        The weighted sum of every entry in loss_dict whose name appears in
        weight_dict.
    """
    loss = 0.0
    for k, v in loss_dict.items():
        if k in weight_dict:
            loss = loss + v * weight_dict[k]
    return loss


def generalized_box_iou(boxes1, boxes2):
    """Compute pairwise generalized IoU between two box sets.

    Standard IoU:
        IoU(A, B) = |A intersect B| / |A union B|

    Generalized IoU adds a penalty for the smallest enclosing box C:
        GIoU(A, B) = IoU(A, B) - (|C| - |A union B|) / |C|

    Args:
        boxes1: Tensor of shape [N, 4] in xyxy format.
        boxes2: Tensor of shape [M, 4] in xyxy format.

    Returns:
        Tensor of shape [N, M] containing pairwise GIoU values.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    area1 = _box_area_xyxy(boxes1)
    area2 = _box_area_xyxy(boxes2)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-6)

    # smallest enclosing box for the GIoU penalty term
    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]

    return iou - (area_c - union) / (area_c + 1e-6)


class HungarianMatcher(nn.Module):
    """Match DETR predictions to ground-truth objects with Hungarian assignment.

    For each image, builds the pairwise matching cost:

        cost(i, j) = lambda_cls * cost_class(i, j)
                   + lambda_bbox * ||b_i - b_j||_1
                   + lambda_giou * (-GIoU(b_i, b_j))

    and solves a minimum-cost bipartite matching so each target is assigned
    to at most one prediction query.
    """

    def __init__(self, cost_class=1.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Return matched prediction/target indices for each batch element.

        Args:
            outputs: Dict containing pred_logits and pred_boxes from DETR.
            targets: List of dicts, one per image, each with labels and boxes.

        Returns:
            List of (src_idx, tgt_idx) tuples. For image b, query src_idx[k]
            is matched to target tgt_idx[k].
        """
        bs, _ = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].softmax(-1)
        out_bbox = outputs["pred_boxes"]

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_bbox = targets[b]["boxes"]
            if tgt_bbox.numel() == 0:
                indices.append(
                    (
                        torch.empty((0,), dtype=torch.int64),
                        torch.empty((0,), dtype=torch.int64),
                    )
                )
                continue
            # Classification cost: negate the predicted probability for each target
            # class. Index out_prob[b] by tgt_ids to select the right columns — result
            # is [Q, T]. Higher predicted probability → lower cost.
            cost_class = -out_prob[b][:, tgt_ids]
            # L1 box cost: torch.cdist with p=1 gives pairwise L1 distances [Q, T].
            cost_bbox = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            # GIoU cost: negate it so minimizing cost means maximizing GIoU.
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]), box_cxcywh_to_xyxy(tgt_bbox)
            )
            C = (
                self.cost_bbox * cost_bbox
                + self.cost_class * cost_class
                + self.cost_giou * cost_giou
            )
            # linear_sum_assignment is from scipy and needs a CPU array.
            C = C.cpu()
            src_ind, tgt_ind = linear_sum_assignment(C)
            indices.append(
                (
                    torch.as_tensor(src_ind, dtype=torch.int64),
                    torch.as_tensor(tgt_ind, dtype=torch.int64),
                )
            )
        return indices


class DETRSetCriterion(nn.Module):
    """Compute the DETR set prediction losses for MP4.

    DETR first finds a one-to-one matching between predicted queries and
    ground-truth objects, then computes losses on the matched pairs.

    Three loss terms:

    1. Classification:
       L_cls = sum_q CE(y_q, y_hat_q)
       Unmatched queries are trained against an extra "no object" class.

    2. Box regression:
       L_bbox = (1 / N) * sum_(q in matched) ||b_q - b_hat_q||_1
       where b_q is the predicted box and b_hat_q is the GT box matched to it.

    3. GIoU loss:
       L_giou = (1 / N) * sum_(q in matched) (1 - GIoU(b_q, b_hat_q))

    The final weighted loss is formed outside this class via compute_total_loss.
    N is the number of target boxes in the batch.

    Refer to the original DETR paper (https://arxiv.org/abs/2005.12872).
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        eos_coef=0.1,
    ):
        """
        Args:
            num_classes: Number of foreground classes.
            matcher: HungarianMatcher used to align queries with targets.
            weight_dict: Dictionary of scalar weights for each named loss.
            eos_coef: Relative weight for the "no object" class in CE loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        # class-weight vector for cross-entropy; last entry is the no-object class
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(self, indices):
        """Flatten per-image matched query indices into batch/query index tensors.

        Returns (batch_idx, src_idx) so that tensor[batch_idx, src_idx] selects
        all matched predictions across the whole batch.

        For example, if we have a batch of 2 images and the matcher returns:
        indices = [
            (tensor([0, 2]), tensor([1, 0])),  # For image 0, query 0 is matched
                                               # to target 1, and query 2 is matched to target 0
            (tensor([1]), tensor([0])),        # For image 1, query 1 is matched to target 0
        ]
        Then the output of _get_src_permutation_idx would be:

        batch_idx = tensor([0, 0, 1])  # Indicates the batch index for each matched query
        src_idx = tensor([0, 2, 1])  # Indicates the query index for each matched query

        This means that for the first image (batch index 0), query indices 0 and 2
        are matched, and for the second image (batch index 1), query index 1 is matched.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(self, outputs, targets, indices):
        """Cross-entropy classification loss over all queries.

        Matched queries get their ground-truth foreground label; unmatched
        queries are trained against the no-object class (index num_classes).

        Args:
            outputs: Dict with pred_logits [B, Q, C+1] and pred_boxes [B, Q, 4].
            targets: List of dicts with labels [T_b] and boxes [T_b, 4].
            indices: List of (src_idx, tgt_idx) pairs from the matcher.

        Returns:
            Dict {"loss_ce": scalar}.
        """
        # TODO: Use _get_src_permutation_idx to get the batch and query indices of
        # the matched predictions.

        # TODO: Construct a tensor containing the GT class label for every matched query.

        # TODO: Fill every query with the no-object label (num_classes) to start, then
        # overwrite just the matched positions with their actual GT class.

        # TODO: Compute CE loss, weighting the no-object class with self.empty_weight.
        # Sum the log probs over the queries and take the mean over the batch.
        # Using F.cross_entropy with default arguments may not give the correct implementation.

        return {"loss_ce": ...}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 and GIoU losses on matched query/target pairs only.

        L_iou = (1 / N) * sum_(q in matched) (1 - GIoU(b_q, b_hat_q))
        L_L1 = (1 / N) * sum_(q in matched) ||b_q - b_hat_q||_1

        where b_q is the predicted box and b_hat_q is the GT box matched to it.

        Args:
            outputs: Dict with pred_logits [B, Q, C+1] and pred_boxes [B, Q, 4],
                where B is the batch size and Q is the number of queries.
            targets: List of dicts with labels [T_b] and boxes [T_b, 4], where
                T_b is the number of target boxes in the batch.
            indices: List of (src_idx, tgt_idx) pairs from the matcher.
            num_boxes: Total number of target boxes in the batch (normalization).

        Returns:
            Dict with scalar tensors loss_bbox and loss_giou.
        """
        # TODO: Use _get_src_permutation_idx to get the batch and query indices of
        # the matched predictions.

        # TODO: Construct a tensor containing the GT class label for every matched query.

        # TODO: Compute the L1 box loss.

        # TODO: Compute the GIoU box loss. You can use the provided generalized_box_iou
        # function, but remember to convert boxes from cxcywh to xyxy format first. Note:
        # The (i, j) entry of the giou matrix contains the GIoU value between src_boxes[i]
        # and target_boxes[j]. We want to sum over the matched pairs, already aligned
        # by Hungarian matching.

        return {"loss_bbox": ..., "loss_giou": ...}

    def forward(self, outputs, targets):
        """Match predictions to targets and compute all DETR loss terms.

        Args:
            outputs: Dict with pred_logits [B, Q, C+1] and pred_boxes [B, Q, 4].
            targets: List of dicts with labels [T_b] and boxes [T_b, 4].

        Returns:
            Dict with loss_ce, loss_bbox, and loss_giou.
        """
        # TODO: Use the Hungarian matcher (self.matcher) to find the best matching
        # between predicted queries and GT targets.

        # TODO: Compute all the requested losses and combine them in a dictionary. The
        # keys should match those in self.weight_dict so they can be combined with compute_total_loss.
        losses = {}
        return losses
