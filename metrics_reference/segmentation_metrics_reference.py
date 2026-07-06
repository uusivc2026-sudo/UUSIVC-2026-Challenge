"""Reference segmentation metrics for UUSIVC2026.

This file is a standalone reference copy of the segmentation metric logic.
It is not imported by the training pipeline.

Project usage locations:
- utils/metrics.py: compute_binary_seg_score_np, compute_binary_seg_score_from_logits,
  compute_ceus_official_score_from_logits
- trainers/trainer_stage1_seg.py: validation calls compute_binary_seg_score_from_logits
  and compute_ceus_official_score_from_logits
- test.py: local segmentation evaluation calls the same project metric functions

Metric definition used by this baseline:
    score = 0.7 * DSC + 0.3 * NSD
where NSD is a lightweight boundary score with 1-pixel tolerance.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


SEG_DSC_WEIGHT = 0.7
SEG_NSD_WEIGHT = 0.3


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 0.0) -> float:
    """Compute binary Dice similarity coefficient."""
    pred = (np.asarray(pred) > 0).astype(np.float32).reshape(-1)
    target = (np.asarray(target) > 0).astype(np.float32).reshape(-1)
    intersection = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    if denom == 0 and eps == 0:
        return 1.0
    return float((2.0 * intersection + eps) / (denom + eps))


def _inner_boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask) > 0
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return center & (~eroded)


def _dilate_cross(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask) > 0
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )


def normalized_surface_dice(pred: np.ndarray, target: np.ndarray, tolerance: int = 1) -> float:
    """Compute the baseline's lightweight binary NSD with 1-pixel tolerance."""
    if tolerance != 1:
        raise ValueError("This reference implementation expects tolerance=1.")

    pred = (np.asarray(pred) > 0).astype(np.uint8)
    target = (np.asarray(target) > 0).astype(np.uint8)

    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    boundary_pred = _inner_boundary(pred)
    boundary_true = _inner_boundary(target)

    if boundary_pred.sum() == 0 and boundary_true.sum() == 0:
        return 1.0
    if boundary_pred.sum() == 0 or boundary_true.sum() == 0:
        return 0.0

    true_match = (boundary_true & _dilate_cross(boundary_pred)).sum()
    pred_match = (boundary_pred & _dilate_cross(boundary_true)).sum()
    return float((true_match + pred_match) / (boundary_true.sum() + boundary_pred.sum()))


def segmentation_score(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Return DSC, NSD, and weighted segmentation score."""
    dsc = dice_score(pred, target, eps=0.0)
    nsd = normalized_surface_dice(pred, target, tolerance=1)
    score = SEG_DSC_WEIGHT * dsc + SEG_NSD_WEIGHT * nsd
    return {"dsc": float(dsc), "nsd": float(nsd), "score": float(score)}
