"""Reference classification metrics for UUSIVC2026.

This file is a standalone reference copy of the classification metric logic.
It is not imported by the training pipeline.

Project usage locations:
- utils/auc_utils.py: binary_accuracy_from_logits, binary_auc_from_logits
- trainers/trainer_stage2_cls.py: validation computes ACC, AUC, and score
- test.py: local classification evaluation computes the same metrics

Metric definition used by this baseline:
    score = 0.5 * ACC + 0.5 * AUC
where AUC is computed from positive-class softmax scores.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def binary_accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary classification accuracy from unnormalized logits."""
    logits = np.asarray(logits)
    labels = np.asarray(labels).astype(np.int64)
    preds = np.argmax(logits, axis=1)
    return float((preds == labels).mean())


def binary_auc_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Compute binary AUC from positive-class softmax scores."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)

    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    prob = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    pos_scores = prob[:, 1]

    pos = pos_scores[labels == 1]
    neg = pos_scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    comparisons = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    auc = (comparisons + 0.5 * ties) / (len(pos) * len(neg))
    return float(auc)


def classification_score(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Return ACC, AUC, and weighted classification score."""
    acc = binary_accuracy_from_logits(logits, labels)
    auc = binary_auc_from_logits(logits, labels)
    score = 0.5 * (acc + auc)
    return {"acc": float(acc), "auc": float(auc), "score": float(score)}
