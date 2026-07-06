# models/framework/unified_model.py

from typing import Dict, Any
import torch
import torch.nn as nn

from models.task_defs import (
    TASK_TYPE,
    IMAGE_SEG,
    IMAGE_CLS,
    VIDEO_SEG,
    VIDEO_CLS,
)

from .image_model import ImageTaskModel
from .video_model import VideoTaskModel


class UnifiedModel(nn.Module):
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
        batch["image"]
        batch["task_type"]

    The organizers keep this baseline note in English for public release.
    {
        "seg_logits": Tensor or None,
        "cls_logits": Tensor or None,
        "features": dict or None,
        "task_type": str,
    }
    """

    def __init__(
        self,
        image_model: ImageTaskModel,
        video_model: VideoTaskModel,
    ):
        super().__init__()
        self.image_model = image_model
        self.video_model = video_model

    @staticmethod
    def _parse_task_type(batch: Dict[str, Any]) -> str:
        if TASK_TYPE not in batch:
            raise KeyError(f"Missing required key: '{TASK_TYPE}'")

        task = batch[TASK_TYPE]

        if isinstance(task, str):
            return task

        if isinstance(task, (list, tuple)):
            if len(task) == 0:
                raise ValueError("Empty task_type list in batch.")
            task0 = task[0]
            if not all(t == task0 for t in task):
                raise ValueError(f"Mixed task types in one batch are not supported: {task}")
            return task0

        raise TypeError(f"Unsupported task_type type: {type(task)}")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        task_type = self._parse_task_type(batch)

        if task_type in {IMAGE_SEG, IMAGE_CLS}:
            outputs = self.image_model(batch)
        elif task_type in {VIDEO_SEG, VIDEO_CLS}:
            outputs = self.video_model(batch)
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

        outputs["task_type"] = task_type
        return outputs