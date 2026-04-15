import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ViTBackbone


def pad_images_to_batch(
    images,
    pad_size_multiple=1,
):
    """Pad a list of images into a batch tensor and padding mask.

    Args:
        images: Either a single tensor [B, C, H, W] or a non-empty list/tuple
            of image tensors [C, H, W].
        pad_size_multiple: If greater than 1, pad height/width up to the nearest
            multiple of this value. Many backbones require divisible spatial sizes.

    Returns:
        Tuple (batch, mask) where:
        - batch has shape [B, C, H_max, W_max]
        - mask has shape [B, H_max, W_max] and uses True for padded
          pixels and False for real image content
    """
    if torch.is_tensor(images):
        return images, None
    if not isinstance(images, (list, tuple)) or len(images) == 0:
        raise TypeError("images must be a Tensor or a non-empty list/tuple of Tensors")
    if any((not torch.is_tensor(im) or im.ndim != 3) for im in images):
        raise ValueError("each image must be a (C,H,W) Tensor")

    max_h = max(int(im.shape[-2]) for im in images)
    max_w = max(int(im.shape[-1]) for im in images)
    pad_size_multiple = int(pad_size_multiple)
    if pad_size_multiple < 1:
        raise ValueError("pad_size_multiple must be >= 1")
    if pad_size_multiple > 1:
        max_h = int(math.ceil(max_h / pad_size_multiple) * pad_size_multiple)
        max_w = int(math.ceil(max_w / pad_size_multiple) * pad_size_multiple)
    batch_size = len(images)

    c = int(images[0].shape[0])
    dtype = images[0].dtype
    device = images[0].device

    batch = torch.zeros((batch_size, c, max_h, max_w), dtype=dtype, device=device)
    mask = torch.ones((batch_size, max_h, max_w), dtype=torch.bool, device=device)
    for i, im in enumerate(images):
        _, h, w = im.shape
        # copy each image into the top-left corner; mark real pixels as False
        batch[i, :, :h, :w] = im
        mask[i, :h, :w] = False

    return batch, mask


class MLP(nn.Module):
    """A simple feedforward network used for bounding-box prediction.

    DETR predicts boxes with a small MLP head rather than a single linear layer.
    The last layer outputs four values interpreted as [cx, cy, w, h].
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class PositionEmbeddingSine(nn.Module):
    """Standard DETR sine-cosine positional encoding for 2D feature maps.

    The encoding assigns each spatial location a deterministic embedding derived
    from its row/column coordinates. This gives the transformer access to spatial
    information after the feature map is flattened into a token sequence.
    """

    def __init__(
        self,
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        scale=None,
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi
        if scale is not None and not normalize:
            raise ValueError("If scale is passed, normalize should be True")

    def forward(self, x, mask):
        """Create a positional encoding aligned with the spatial shape of x.

        Args:
            x: Feature map of shape [B, C, H, W].
            mask: Optional padding mask of shape [B, H, W] where True
                indicates padding.

        Returns:
            Positional encoding tensor of shape [B, 2 * num_pos_feats, H, W].
        """
        b, _, h, w = x.shape
        if mask is None:
            not_mask = torch.ones((b, h, w), device=x.device, dtype=torch.bool)
        else:
            not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).contiguous()
        return pos


class SimpleDETR(nn.Module):
    """A simplified DETR-style detector for the MP4 assignment.

    The model follows the standard DETR pipeline:

    1. Extract a spatial feature map with a CNN or ViT backbone.
    2. Project the backbone features to the transformer hidden size d_model.
    3. Add 2D positional encodings to the flattened feature sequence.
    4. Run a transformer encoder-decoder using learned object queries.
    5. Predict one class distribution and one normalized box per query.

    Outputs:
    - pred_logits: [batch_size, num_queries, num_classes + 1]
    - pred_boxes: [batch_size, num_queries, 4] in normalized [cx, cy, w, h] format
    """

    def __init__(
        self,
        num_classes=20,
        backbone="facebook/dinov2-with-registers-small",
        num_queries=25,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        """
        Args:
            num_classes: Number of foreground object classes.
            backbone: Backbone name passed to ViTBackbone.
            num_queries: Number of learned object queries.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Number of transformer encoder layers.
            num_decoder_layers: Number of transformer decoder layers.
            dim_feedforward: Hidden size of the transformer's feedforward blocks.
            dropout: Dropout probability inside the transformer.
        """
        super().__init__()
        self.backbone = ViTBackbone(name=backbone)
        self.position_embedding = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.num_queries = num_queries

        # TODO: self.input_proj is a 1x1 convolution that projects the backbone's output feature
        # dimension to the transformer's hidden dimension d_model.

        # TODO: Initialize self.transformer as a Transformer with the specified hyperparameters.
        # Refer to https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html.

        # TODO: self.query_embed is a learnable embedding of shape [num_queries, d_model].
        # You can use nn.Embedding to initialize it.

        # TODO: self.class_embed is a linear layer that projects the transformer's
        # decoder output to class logits. Account for the extra "no object" class.

        # TODO: self.bbox_embed is a 3-layer MLP that projects the decoder output
        # to 4 box coordinates. Use the provided MLP class.

    def forward(
        self,
        images,
    ):
        """Run the simplified DETR model on a batch of images.

        Args:
            images: Either a tensor of shape [B, C, H, W] or a list of
                [C, H, W] tensors.
            mask: Optional boolean padding mask of shape [B, H, W] where
                True marks padded pixels.

        Returns:
            Dictionary with pred_logits and pred_boxes.
        """
        input_size_multiple = getattr(self.backbone, "input_size_multiple", 1)
        # pad variable-size images into a single batch; mask tracks padding
        x, mask = pad_images_to_batch(images, pad_size_multiple=input_size_multiple)

        # TODO: Extract a feature map from the backbone

        # TODO: Project the backbone features to the transformer hidden size
        src = ...

        # The padding mask is at full image resolution but src is spatially smaller.
        # Use F.interpolate to resize it.
        mask = (
            F.interpolate(mask.unsqueeze(1).float(), size=src.shape[-2:])
            .to(dtype=torch.bool)
            .squeeze(1)
        )

        # TODO: Get a positional encoding using src and the mask.

        # TODO: The backbone gives [B, C, H, W] but the transformer expects [B, seq_len, C].
        # You will need to flatten the spatial dimensions and transpose. Make sure
        # the positional encoding and the mask are processed the same way.

        # TODO: Expand query_embed to the batch dimension so every image gets the same
        # set of learned queries.

        # TODO: src with positional encoding goes into the encoder, while query
        # embeddings go into the decoder. Ensure appropriate masking.

        # TODO: Project the decoder output to class logits.

        # TODO: Project to box coordinates, then apply sigmoid to keep
        # predictions in [0, 1]. Without it the model can predict boxes outside the image.

        return {
            "pred_logits": ...,
            "pred_boxes": ...,
        }
