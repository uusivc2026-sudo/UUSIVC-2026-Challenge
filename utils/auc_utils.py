from typing import Iterable

import numpy as np


def binary_accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float((preds == labels).mean())


def binary_auc_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    prob = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    pos_scores = prob[:, 1]
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos = pos_scores[pos_mask]
    neg = pos_scores[neg_mask]

    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    comparisons = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    auc = (comparisons + 0.5 * ties) / (len(pos) * len(neg))
    return float(auc)


def stack_numpy(items: Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(x) for x in items], axis=0)
