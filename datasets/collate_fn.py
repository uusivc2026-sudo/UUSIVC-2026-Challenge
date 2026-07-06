# datasets/collate_fn.py

from typing import List, Dict, Any
import torch

from .task_defs import (
    IMAGE,
    LABEL_CLS,
    LABEL_SEG,
    TASK_TYPE,
    MODALITY,
    DATASET_NAME,
    IS_VIDEO,
    NUM_FRAMES,
    PATIENT_ID,
    CASE_ID,
    POSITION_PROMPT,
    TASK_PROMPT,
    MODE_PROMPT,
    TYPE_PROMPT,
)


VISUAL_IMAGE = "visual_image"
VISUAL_GT_MASK = "visual_gt_mask"
OFFICIAL_GT_MASK = "official_gt_mask"
CEUS_RESTORE_META = "ceus_restore_meta"


def _stack_if_not_none(items):
    """
    items: list of tensors or None
    The organizers keep this baseline note in English for public release.
    """
    if len(items) == 0:
        return None
    if items[0] is None:
        return None
    return torch.stack(items, dim=0)


def _collect_optional(batch: List[Dict[str, Any]], key: str):
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    """
    if key not in batch[0]:
        return None

    values = [x.get(key, None) for x in batch]
    if values[0] is None:
        return None

    if torch.is_tensor(values[0]):
        return torch.stack(values, dim=0)

    return values


def collate_image_task_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
    - image: [C,H,W] -> [B,C,H,W]
    - label_cls: int -> [B]
    The organizers keep this baseline note in English for public release.
    """
    output = {IMAGE: torch.stack([x[IMAGE] for x in batch], dim=0),
              LABEL_CLS: torch.tensor([x[LABEL_CLS] for x in batch], dtype=torch.long),
              LABEL_SEG: _stack_if_not_none([x[LABEL_SEG] for x in batch]), TASK_TYPE: [x[TASK_TYPE] for x in batch],
              MODALITY: [x[MODALITY] for x in batch], DATASET_NAME: [x[DATASET_NAME] for x in batch],
              IS_VIDEO: torch.tensor([x[IS_VIDEO] for x in batch], dtype=torch.bool),
              NUM_FRAMES: torch.tensor([x[NUM_FRAMES] for x in batch], dtype=torch.long),
              PATIENT_ID: [x.get(PATIENT_ID, None) for x in batch], CASE_ID: [x.get(CASE_ID, None) for x in batch],
              POSITION_PROMPT: _collect_optional(batch, POSITION_PROMPT),
              TASK_PROMPT: _collect_optional(batch, TASK_PROMPT), MODE_PROMPT: _collect_optional(batch, MODE_PROMPT),
              TYPE_PROMPT: _collect_optional(batch, TYPE_PROMPT),
              VISUAL_IMAGE: _collect_optional(batch, VISUAL_IMAGE),
              VISUAL_GT_MASK: _collect_optional(batch, VISUAL_GT_MASK),
              OFFICIAL_GT_MASK: _collect_optional(batch, OFFICIAL_GT_MASK),
              CEUS_RESTORE_META: _collect_optional(batch, CEUS_RESTORE_META)}

    # The organizers keep this baseline step explicit for participants.

    return output


def collate_video_task_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
    - image: [T,C,H,W] -> [B,T,C,H,W]
    - label_cls: int -> [B]
    The organizers keep this baseline note in English for public release.
    """
    output = {IMAGE: torch.stack([x[IMAGE] for x in batch], dim=0),
              LABEL_CLS: torch.tensor([x[LABEL_CLS] for x in batch], dtype=torch.long),
              LABEL_SEG: _stack_if_not_none([x[LABEL_SEG] for x in batch]), TASK_TYPE: [x[TASK_TYPE] for x in batch],
              MODALITY: [x[MODALITY] for x in batch], DATASET_NAME: [x[DATASET_NAME] for x in batch],
              IS_VIDEO: torch.tensor([x[IS_VIDEO] for x in batch], dtype=torch.bool),
              NUM_FRAMES: torch.tensor([x[NUM_FRAMES] for x in batch], dtype=torch.long),
              PATIENT_ID: [x.get(PATIENT_ID, None) for x in batch], CASE_ID: [x.get(CASE_ID, None) for x in batch],
              POSITION_PROMPT: _collect_optional(batch, POSITION_PROMPT),
              TASK_PROMPT: _collect_optional(batch, TASK_PROMPT), MODE_PROMPT: _collect_optional(batch, MODE_PROMPT),
              TYPE_PROMPT: _collect_optional(batch, TYPE_PROMPT),
              VISUAL_IMAGE: _collect_optional(batch, VISUAL_IMAGE),
              VISUAL_GT_MASK: _collect_optional(batch, VISUAL_GT_MASK),
              OFFICIAL_GT_MASK: _collect_optional(batch, OFFICIAL_GT_MASK),
              CEUS_RESTORE_META: _collect_optional(batch, CEUS_RESTORE_META)}

    # The organizers keep this baseline step explicit for participants.

    return output
