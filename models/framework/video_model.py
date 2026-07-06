# models/framework/video_model.py

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from models.task_defs import (
    IMAGE,
    TASK_TYPE,
    VIDEO_SEG,
    VIDEO_CLS,
    POSITION_PROMPT,
    TASK_PROMPT,
    MODE_PROMPT,
    TYPE_PROMPT,
)


class VideoTaskModel(nn.Module):
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
        backbone.forward_video_features(batch, prompts=None) -> feat_dict

    The organizers keep this baseline note in English for public release.
    {
        "encoder_out": ...,
        "skip_features": ...,
        "seg_feat": Tensor,   # [B, T, C, H, W]
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
    }

    The organizers keep this baseline note in English for public release.
        temporal_router(
            task_type=task_type,
            feat_dict=feat_dict,
        ) -> {
            "seg_logits": ... or None,
            "cls_logits": ... or None,
            "features": feat_dict or richer dict
        }
    """

    def __init__(
        self,
        backbone: nn.Module,
        temporal_router: nn.Module,
        use_prompt: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.temporal_router = temporal_router
        self.use_prompt = use_prompt

    def _extract_prompts(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.use_prompt:
            return None

        return {
            "position_prompt": batch.get(POSITION_PROMPT, None),
            "task_prompt": batch.get(TASK_PROMPT, None),
            "mode_prompt": batch.get(MODE_PROMPT, None),
            "type_prompt": batch.get(TYPE_PROMPT, None),
        }

    @staticmethod
    def _parse_task_type(batch: Dict[str, Any]) -> str:
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
        if IMAGE not in batch:
            raise KeyError(f"Missing required key: '{IMAGE}'")

        task_type = self._parse_task_type(batch)
        if task_type not in {VIDEO_SEG, VIDEO_CLS}:
            raise ValueError(f"VideoTaskModel only supports video tasks, got {task_type}")

        prompts = self._extract_prompts(batch)

        feat_dict = self.backbone.forward_video_features(batch, prompts=prompts)

        outputs = self.temporal_router(
            task_type=task_type,
            feat_dict=feat_dict,
        )

        if "seg_logits" not in outputs:
            outputs["seg_logits"] = None
        if "cls_logits" not in outputs:
            outputs["cls_logits"] = None
        if "features" not in outputs:
            outputs["features"] = feat_dict

        return outputs