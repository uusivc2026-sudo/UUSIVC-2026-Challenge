# models/framework/image_model.py

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from models.task_defs import (
    IMAGE,
    TASK_TYPE,
    IMAGE_SEG,
    IMAGE_CLS,
    POSITION_PROMPT,
    TASK_PROMPT,
    MODE_PROMPT,
    TYPE_PROMPT,
)


class ImageTaskModel(nn.Module):
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
        backbone.forward_image_features(batch, prompts=None) -> feat_dict

    The organizers keep this baseline note in English for public release.
    {
        "encoder_out": ...,
        "skip_features": ...,
        "seg_feat": Tensor,   # [B, C, H, W]
        The organizers keep this baseline note in English for public release.
    }
    """

    def __init__(
        self,
        backbone: nn.Module,
        seg_head: nn.Module,
        cls_head: nn.Module,
        use_prompt: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.seg_head = seg_head
        self.cls_head = cls_head
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

    def forward_image_seg(
        self,
        feat_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        feat_dict["seg_feat"]: [B, C, H, W]
        """
        if "seg_feat" not in feat_dict:
            raise KeyError("backbone.forward_image_features must return key 'seg_feat'")

        seg_feat = feat_dict["seg_feat"]
        seg_logits = self.seg_head(seg_feat)

        return {
            "seg_logits": seg_logits,
            "cls_logits": None,
            "features": feat_dict,
        }

    def forward_image_cls(
        self,
        feat_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        feat_dict["cls_feat"]: [B, C] or [B, L, C]
        """
        if "cls_feat" not in feat_dict:
            raise KeyError("backbone.forward_image_features must return key 'cls_feat'")

        cls_feat = feat_dict["cls_feat"]
        cls_logits = self.cls_head(cls_feat)

        return {
            "seg_logits": None,
            "cls_logits": cls_logits,
            "features": feat_dict,
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if IMAGE not in batch:
            raise KeyError(f"Missing required key: '{IMAGE}'")

        task_type = self._parse_task_type(batch)
        if task_type not in {IMAGE_SEG, IMAGE_CLS}:
            raise ValueError(f"ImageTaskModel only supports image tasks, got {task_type}")

        prompts = self._extract_prompts(batch)

        # The organizers keep this baseline step explicit for participants.
        feat_dict = self.backbone.forward_image_features(batch, prompts=prompts)

        if task_type == IMAGE_SEG:
            return self.forward_image_seg(feat_dict)
        elif task_type == IMAGE_CLS:
            return self.forward_image_cls(feat_dict)

        raise ValueError(f"Unsupported task_type in ImageTaskModel: {task_type}")