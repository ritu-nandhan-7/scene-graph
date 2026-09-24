"""Pair feature extraction: union crop, 8-D geometry and 512-D visual feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
from PIL import Image

from datasets.visual_genome import BoundingBox

# Reuse the exact union/crop/normalization helpers that produced the
# cached training features (expansion = 0.20).
from notebook.models.relationship.geometry import (
    bbox_relative_to_crop,
    crop_union_region,
    get_union_bbox,
    normalize_bbox,
)


@dataclass
class PairFeatures:
    """Feature bundle for one (subject, object) pair, ready for the model."""

    visual: torch.Tensor  # (1, 512) float32
    geometry: torch.Tensor  # (1, 8) float32, normalized within union crop
    union_crop: Image.Image


@dataclass
class PairBatchFeatures:
    """Feature bundle for a whole batch of (subject, object) pairs.

    ``visual[i]`` / ``geometry[i]`` belong to ``pairs[i]`` and are numerically
    identical to what :func:`extract_pair_features` would return for that pair:
    the crop, the preprocessing and the geometry maths are the same code, only
    the CLIP call is shared.
    """

    visual: torch.Tensor  # (N, 512) float32
    geometry: torch.Tensor  # (N, 8) float32
    union_boxes: List[BoundingBox]



def build_geometry(
    subject_bbox: BoundingBox,
    object_bbox: BoundingBox,
    union_box: BoundingBox,
) -> torch.Tensor:
    """8-D normalized geometry: [sub cx, cy, w, h, obj cx, cy, w, h]."""
    subject_relative = bbox_relative_to_crop(subject_bbox, union_box)
    object_relative = bbox_relative_to_crop(object_bbox, union_box)

    subject = normalize_bbox(subject_relative, union_box)
    obj = normalize_bbox(object_relative, union_box)

    return torch.tensor(
        [
            subject.cx,
            subject.cy,
            subject.width,
            subject.height,
            obj.cx,
            obj.cy,
            obj.width,
            obj.height,
        ],
        dtype=torch.float32,
    )


def build_union_crop(
    image: Image.Image,
    subject_bbox: BoundingBox,
    object_bbox: BoundingBox,
    expansion: float = 0.20,
):
    """Expanded union bounding box and its crop from the image."""
    union_box = get_union_bbox(subject_bbox, object_bbox, expansion=expansion)
    return union_box, crop_union_region(image, union_box)


def extract_pair_features(
    image: Image.Image,
    subject_bbox: BoundingBox,
    object_bbox: BoundingBox,
    visual_encoder,
    device: str = "cpu",
    expansion: float = 0.20,
) -> PairFeatures:
    """Extract the exact inputs the trained model expects for one pair.

    Pipeline (identical to notebook 05):

        union crop (expansion 0.20)
        -> open_clip preprocess (224 x 224)
        -> frozen CLIP ViT-B/32 -> (1, 512) visual feature
        + 8-D normalized geometry within the union crop
    """
    union_box, crop = build_union_crop(image, subject_bbox, object_bbox, expansion)
    geometry = build_geometry(subject_bbox, object_bbox, union_box)

    visual_input = visual_encoder.preprocess(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        visual = visual_encoder(visual_input)  # (1, 512)

    return PairFeatures(
        visual=visual,
        geometry=geometry.unsqueeze(0).to(device),
        union_crop=crop,
    )


def extract_pair_features_batch(
    image: Image.Image,
    pairs: Sequence[Tuple[BoundingBox, BoundingBox]],
    visual_encoder,
    device: str = "cpu",
    expansion: float = 0.20,
    batch_size: int = 16,
) -> PairBatchFeatures:
    """Batched version of :func:`extract_pair_features` for many pairs.

    For every pair exactly the same work is done as in the single-pair path
    (expanded union crop -> geometry -> open_clip preprocessing), but the CLIP
    forward pass is executed once per ``batch_size`` crops instead of once per
    pair.  This is a pure scheduling change: the model, the weights, the
    preprocessing and the resulting 512-D features are unchanged, so predictions
    are identical to the per-pair path.

    Returns one row per pair, in input order.  An empty ``pairs`` sequence
    yields correctly shaped empty tensors.
    """
    visual_chunks: List[torch.Tensor] = []
    geometry_rows: List[torch.Tensor] = []
    union_boxes: List[BoundingBox] = []
    pending: List[torch.Tensor] = []

    def flush() -> None:
        if not pending:
            return
        batch = torch.stack(pending, dim=0).to(device)
        # visual_encoder already runs under no_grad; the explicit context keeps
        # this path grad-free even if a different encoder is injected.
        with torch.no_grad():
            visual_chunks.append(visual_encoder(batch).float().cpu())
        pending.clear()

    for subject_bbox, object_bbox in pairs:
        union_box, crop = build_union_crop(image, subject_bbox, object_bbox, expansion)
        union_boxes.append(union_box)
        geometry_rows.append(build_geometry(subject_bbox, object_bbox, union_box))
        pending.append(visual_encoder.preprocess(crop))
        if len(pending) >= max(1, batch_size):
            flush()
    flush()

    if visual_chunks:
        visual = torch.cat(visual_chunks, dim=0)
    else:
        visual = torch.empty((0, 512), dtype=torch.float32)
    if geometry_rows:
        geometry = torch.stack(geometry_rows, dim=0)
    else:
        geometry = torch.empty((0, 8), dtype=torch.float32)

    return PairBatchFeatures(
        visual=visual,
        geometry=geometry.to(device),
        union_boxes=union_boxes,
    )