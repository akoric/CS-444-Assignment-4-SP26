import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Image
 └── Grid (S × S)
      └── Each cell
           └── B box predictions
                └── each box = (x, y, w, h, confidence)
"""

def compute_iou(boxes1_xyxy, boxes2_xyxy):
    """Compute pairwise IoU for two box sets in [x1, y1, x2, y2] format.

    Args:
        boxes1_xyxy: Tensor of shape [N, 4].
        boxes2_xyxy: Tensor of shape [M, 4].

    Returns:
        Tensor of shape [N, M] containing pairwise IoUs.
    """
    num_boxes1 = boxes1_xyxy.size(0)
    num_boxes2 = boxes2_xyxy.size(0)

    top_left = torch.max(
        boxes1_xyxy[:, :2].unsqueeze(1).expand(num_boxes1, num_boxes2, 2),
        boxes2_xyxy[:, :2].unsqueeze(0).expand(num_boxes1, num_boxes2, 2),
    )
    bottom_right = torch.min(
        boxes1_xyxy[:, 2:].unsqueeze(1).expand(num_boxes1, num_boxes2, 2),
        boxes2_xyxy[:, 2:].unsqueeze(0).expand(num_boxes1, num_boxes2, 2),
    )

    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection_area = intersection_wh[:, :, 0] * intersection_wh[:, :, 1]

    area1 = (boxes1_xyxy[:, 2] - boxes1_xyxy[:, 0]) * (
        boxes1_xyxy[:, 3] - boxes1_xyxy[:, 1]
    )
    area2 = (boxes2_xyxy[:, 2] - boxes2_xyxy[:, 0]) * (
        boxes2_xyxy[:, 3] - boxes2_xyxy[:, 1]
    )
    union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - intersection_area

    return intersection_area / union_area.clamp(min=1e-6)


class YOLOLoss(nn.Module):
    """YOLO-style detection loss for the MP4 object detector.

    Each grid cell predicts B boxes followed by C class scores:

    [x, y, w, h, confidence] * B + [class_1, ..., class_C]

    The loss has three terms (see Lecture 12):

    1. Regression:
       lambda_coord * 1_ij^obj * ((x - x_hat)^2 + (y - y_hat)^2)
       lambda_coord * 1_ij^obj * ((sqrt(w) - sqrt(w_hat))^2 + (sqrt(h) - sqrt(h_hat))^2)
    2. Object / no-object confidence:
       1_ij^obj * (C - C_hat)^2 + lambda_noobj * 1_ij^noobj * (C - C_hat)^2
    3. Class prediction:
       1_i^obj * sum_c (p_i(c) - p_hat_i(c))^2

    1_ij^obj = 1 for the single predictor j in cell i that has the highest IoU
    with the ground-truth box.
    """

    def __init__(self, grid_size, boxes_per_cell, lambda_coord, lambda_noobj):
        """
        Args:
            grid_size: Number of cells per spatial dimension (S in the slide).
            boxes_per_cell: Number of box predictors per cell (B in the slide).
            lambda_coord: Weight on the box regression terms.
            lambda_noobj: Weight on the no-object confidence term.
        """
        super().__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def split_prediction_tensor(self, prediction_tensor):
        """Split the raw model output into box predictions and class scores.

        Args:
            prediction_tensor: Tensor of shape [N, S, S, B * 5 + C].

        Returns:
            (predicted_boxes, predicted_class_scores) where predicted_boxes has
            shape [N, S, S, B, 5] (each row is [x, y, w, h, confidence]) and
            predicted_class_scores has shape [N, S, S, C].
        """
        N = prediction_tensor.shape[0]
        S = prediction_tensor.shape[1]
        B = self.boxes_per_cell

        box_part = prediction_tensor[..., :B * 5] # first B*5 values
        predicted_boxes = box_part.view(N, S, S, B, 5) #  unpacks those into B predictors, each with 5 values [x, y, w, h, conf]
        predicted_class_scores = prediction_tensor[..., B * 5:] # the remaining C values → the class scores for that cell

        return predicted_boxes, predicted_class_scores
        

    def xywh_to_xyxy(self, boxes_xywh, col_row):
        """Convert boxes from center-size form to corner form.

        Args:
            boxes_xywh: Tensor of shape [N, 4] with boxes as
                [x_center, y_center, width, height], where x and y are
                within-cell offsets in [0, 1].
            col_row: Tensor of shape [N, 2] containing the (col, row) grid
                index of each box's cell, used to recover the absolute
                image-normalized center via (col + x) / S.

        Returns:
            Tensor of shape [N, 4] with boxes as [x1, y1, x2, y2] in
            image-normalized coordinates.

        Example:
            A box in cell (col=3, row=2) with offset (0.5, 0.5) and size
            (0.3, 0.4) (with S=14) becomes center ((3.5)/14, (2.5)/14) and
            corners (3.5/14 - 0.3/2, 2.5/14 - 0.4/2, 3.5/14 + 0.3/2, 2.5/14 + 0.4/2).
        """
        # TODO: Return boxes in corner form from boxes in center-size form.
        # Input x and y are offsets within the cell; col_row gives the cell's grid position.
        
        x_offset = boxes_xywh[:, 0]
        y_offset = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        col = col_row[:, 0]
        row = col_row[:, 1]

        S = self.grid_size

        global_x = (col + x_offset) / S
        global_y = (row + y_offset) / S

        x1 = global_x - w / 2
        y1 = global_y - h / 2
        x2 = global_x + w / 2
        y2 = global_y  + h / 2

        return torch.stack([x1, y1, x2, y2], dim=1)  # shape [N, 4]

    def choose_responsible_box(self, predicted_boxes, target_boxes, has_object_mask):
        """Find the responsible predictor for each cell that contains an object.

        Args:
            predicted_boxes: Tensor of shape [N, S, S, B, 5].
            target_boxes: Tensor of shape [N, S, S, 4].
            has_object_mask: Boolean tensor of shape [N, S, S].

        Returns:
            (responsible_box_predictions, responsible_box_ious,
            responsible_predictor_index) with shapes [num_object_cells, 5],
            [num_object_cells, 1], and [num_object_cells].

        The responsible predictor is whichever of the B predictors has the
        highest IoU with the ground-truth box in that cell.
        """
        # TODO: Use has_object_mask to filter predicted and target boxes to cells
        # that contain objects.

        # Each row = one object cell → B predictions compete against 1 ground-truth box
        obj_predicted_boxes = predicted_boxes[has_object_mask]
        obj_target_boxes = target_boxes[has_object_mask]

        # K = num_object_cells
        K = obj_predicted_boxes.shape[0] 
        B = self.boxes_per_cell

        # TODO: Edge case: no object cells in batch.
        if K == 0:
            empty_preds = predicted_boxes.new_zeros((0, 5))
            empty_ious  = predicted_boxes.new_zeros((0, 1))
            empty_idx   = predicted_boxes.new_zeros((0,), dtype=torch.long)
            return empty_preds, empty_ious, empty_idx


        # TODO: Get (col, row) grid indices for cells with objects, and use
        # self.xywh_to_xyxy to convert predicted and target boxes to corner form
        # in image-normalized coordinates for IoU computation. Loop over the B
        # predictors to compute their IoUs with the GT box and find the max.
        # Note that compute_iou returns an [N, N] matrix of pairwise IoUs, but you
        # only need the part comparing each predictor to its own cell's GT box.
        
        # Extracting all the grid cells where there is an object (True), 
        # and storing their (row, col) indices in a matrix.
        cell_indices = has_object_mask.nonzero(as_tuple=False)
        col = cell_indices[:, 2]
        row = cell_indices[:, 1]
        col_row = torch.stack([col, row], dim=1).float()
        # Convert the ground-truth boxes to corner coordinates
        target_xyxy = self.xywh_to_xyxy(obj_target_boxes, col_row)

        iou_per_predictor = []

        # loops through each of the B predictors in the cel
        for b in range(B):
            pred_b_xywh = obj_predicted_boxes[:, b, :4]
            pred_b_xyxy = self.xywh_to_xyxy(pred_b_xywh, col_row)
            iou_matrix = compute_iou(pred_b_xyxy, target_xyxy)
            iou_diag = iou_matrix.diagonal()
            iou_per_predictor.append(iou_diag)
    
        iou_per_predictor = torch.stack(iou_per_predictor, dim=1)


        # TODO: Return the responsible predictor's box predictions, IoU with the GT
        # box, and predictor index within the cell.
        
        # Choose the best predictor in each cell
        best_ious, best_indices = iou_per_predictor.max(dim=1)
        gather_idx = best_indices.view(K, 1, 1).expand(K, 1, 5)
        responsible_box_predictions = obj_predicted_boxes.gather(1, gather_idx).squeeze(1)
        responsible_box_ious = best_ious.unsqueeze(1) 
        
        return responsible_box_predictions, responsible_box_ious, best_indices
        

    def build_responsible_mask(
        self,
        predicted_boxes,
        has_object_mask,
        responsible_predictor_index,
    ):
        """Build a [N, S, S, B] boolean mask representing 1_ij^obj in the formula.

        The mask is True for the single responsible predictor in each object
        cell (the one with the highest IoU with the ground-truth box) and
        False everywhere else, including all predictors in empty cells.
        """
        # TODO: Create a responsible boolean mask. Remember to account for the edge
        # case where there are no object cells.
        
        obj_predicted_boxes = predicted_boxes[has_object_mask]

        K = obj_predicted_boxes.shape[0] 
        B = self.boxes_per_cell
        S = self.grid_size
        N = predicted_boxes.shape[0]

        mask = torch.zeros((N, S, S, B), dtype=torch.bool, device=predicted_boxes.device)

        if K == 0:
            return mask

        cell_indices = has_object_mask.nonzero(as_tuple=False)
        n = cell_indices[:, 0]
        row = cell_indices[:, 1]
        col = cell_indices[:, 2]

        pred = responsible_predictor_index

        mask[n, row, col, pred] = True

        return mask
            

    def regression_xy_loss(
        self,
        responsible_box_predictions,
        target_boxes_for_object_cells,
    ):
        """Sum of squared errors on (x, y) for the responsible predictor in each object cell.

        sum_i sum_j 1_ij^obj * ((x_i - x_hat_i)^2 + (y_i - y_hat_i)^2)

        Note: The 1_ij^obj factor is implicit, since both inputs have already been
        filtered down to the single responsible predictor in each object cell.
        """
        x = responsible_box_predictions[:, 0]
        y = responsible_box_predictions[:, 1]

        x_hat = target_boxes_for_object_cells[:, 0]
        y_hat = target_boxes_for_object_cells[:, 1]

        xy_loss = ((x - x_hat)**2 + (y - y_hat)**2).sum()

        return xy_loss
 
    def regression_wh_loss(
        self,
        responsible_box_predictions,
        target_boxes_for_object_cells,
    ):
        """Sum of squared errors on sqrt(w), sqrt(h) for the responsible predictor.

        sum_i sum_j 1_ij^obj * ((sqrt(w_i) - sqrt(w_hat_i))^2 +
                                 (sqrt(h_i) - sqrt(h_hat_i))^2)

        Note: The 1_ij^obj factor is implicit, similarly to the xy loss.

        The square root makes the loss scale-invariant: a 2px error on a 10px
        box is penalized more than the same error on a 100px box.
        """
        # TODO: Regress sqrt(w) and sqrt(h). Clamp to 1e-6 before sqrt, since early
        # in training w/h can go slightly negative, and sqrt of a negative number gives NaN.

        w = responsible_box_predictions[:, 2]
        w_clamped = torch.clamp(w, min=1e-6)
        sqrt_w = torch.sqrt(w_clamped)


        h = responsible_box_predictions[:, 3]
        h_clamped = torch.clamp(h, min=1e-6)
        sqrt_h = torch.sqrt(h_clamped)

        w_hat = target_boxes_for_object_cells[:, 2]
        w_hat_clamped = torch.clamp(w_hat, min=1e-6)
        sqrt_w_hat = torch.sqrt(w_hat_clamped)

        h_hat = target_boxes_for_object_cells[:, 3]
        h_hat_clamped = torch.clamp(h_hat, min=1e-6)
        sqrt_h_hat = torch.sqrt(h_hat_clamped)

        wh_loss = ((sqrt_w  - sqrt_w_hat)**2 + (sqrt_h - sqrt_h_hat)**2).sum()

        return wh_loss
        
    def object_confidence_loss(self, responsible_box_predictions, responsible_box_ious):
        """Sum of squared errors on confidence for predictors assigned to real objects.

        sum_i sum_j 1_ij^obj * (C_i - C_hat_i)^2

        The confidence target is the IoU between the predicted box and the
        ground-truth box (detached, so it acts as a fixed regression target).
        """
        # TODO: The confidence target isn't 1.0 for matched boxes; it's the IoU with
        # the GT box. Use responsible_box_ious as the target and detach it; otherwise
        # gradients flow back through the IoU computation and training becomes unstable.
        C = responsible_box_predictions[:, 4]
        C_hat = responsible_box_ious[:, 0].detach()

        obj_loss = ((C - C_hat)**2).sum()

        return obj_loss

    def no_object_confidence_loss(self, predicted_boxes, responsible_mask):
        """Sum of squared errors on confidence for all non-responsible predictors.

        sum_i sum_j 1_ij^noobj * (C_i - C_hat_i)^2

        Covers both empty cells and the losing predictors in object cells.
        The target confidence is 0 for all of these.
        """
        C = predicted_boxes[..., 4]

        # sets True for: empty-cell predictors & losing predictors in object cells
        no_obj_mask = ~responsible_mask

        # C[noobj_mask] 
        # * selects only the confidence values where noobj_mask is True
        # * So it keeps: all predictors in empty cells & all non-responsible predictors in object cells
        
        # loss pushes confidence → 0 for all predictors that should not detect an object
        no_obj_loss = ((C[no_obj_mask] - 0)**2).sum() 

        return no_obj_loss

    def class_probability_loss(
        self,
        predicted_class_scores,
        target_class_scores,
        has_object_mask,
    ):
        """Sum of squared errors in class probabilities, only for cells that contain an object.

        sum_i 1_i^obj * sum_c (p_i(c) - p_hat_i(c))^2

        Class scores are per cell, not per predictor box.
        """
        # Note: don't slide along [:,3] bc full class vector per object cell is needed
        p_c = predicted_class_scores[has_object_mask]
        p_c_hat = target_class_scores[has_object_mask]
        
        class_loss = ((p_c - p_c_hat)**2).sum()

        return class_loss


    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """Compute the full YOLO loss and return each named component.

        Args:
            pred_tensor: Model output of shape [N, S, S, B * 5 + C].
            target_boxes: Ground-truth boxes of shape [N, S, S, 4].
            target_cls: One-hot class targets of shape [N, S, S, C].
            has_object_map: Boolean tensor of shape [N, S, S].

        Returns:
            Dict with total_loss and the individual components: reg_loss,
            reg_xy_loss, reg_wh_loss, obj_loss, no_obj_loss, cls_loss.
        """
        # TODO: Step 1: Split pred_tensor into predicted_boxes and predicted_class_scores.

        # TODO: Step 2: Find the responsible predictor for each object cell and
        # build the responsible mask (1_ij^obj).

        # TODO: Step 3: Compute regression loss terms.

        # TODO: Step 4: Compute confidence loss terms for object and no-object predictors.

        # TODO: Step 5: Compute class probability.

        # TODO: Step 6: Scale the terms by the appropriate weights, and divide
        # every term by batch_size to normalize.

        return {
            "total_loss": ...,
            "reg_loss": ...,
            "reg_xy_loss": ...,
            "reg_wh_loss": ...,
            "obj_loss": ...,
            "no_obj_loss": ...,
            "cls_loss": ...,
        }
