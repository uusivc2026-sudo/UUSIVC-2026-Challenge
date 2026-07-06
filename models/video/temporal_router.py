# models/video/temporal_router.py

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from models.task_defs import VIDEO_SEG, VIDEO_CLS


class TemporalRouter(nn.Module):
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    """

    def __init__(
        self,
        simple_video_seg_head: Optional[nn.Module] = None,
        memory_video_seg_head: Optional[nn.Module] = None,
        video_cls_head: Optional[nn.Module] = None,
        use_memory_seg: bool = False,
        seg_mode: str = "simple",        # "simple" / "memory"
        cls_mode: str = "simple",        # "simple" / "neighbor"
    ):
        super().__init__()
        self.simple_video_seg_head = simple_video_seg_head
        self.memory_video_seg_head = memory_video_seg_head
        self.video_cls_head = video_cls_head

        self.use_memory_seg = use_memory_seg
        self.seg_mode = seg_mode
        self.cls_mode = cls_mode

    # ==========================================================
    # video segmentation
    # ==========================================================
    def forward_video_seg_simple(
        self,
        feat_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        """
        if self.simple_video_seg_head is None:
            raise RuntimeError("simple_video_seg_head is None, cannot run simple video segmentation.")

        if "seg_feat" not in feat_dict:
            raise KeyError("feat_dict missing 'seg_feat' for video segmentation.")

        seg_feat = feat_dict["seg_feat"]
        seg_logits = self.simple_video_seg_head(seg_feat)   # [B, T, K, H, W]

        return {
            "seg_logits": seg_logits,
            "cls_logits": None,
            "features": feat_dict,
        }

    def forward_video_seg_memory(
        self,
        feat_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.

        The organizers keep this baseline note in English for public release.
            seg_logits = memory_video_seg_head(feat_dict)
        """
        if self.memory_video_seg_head is None:
            raise RuntimeError("memory_video_seg_head is None, cannot run memory video segmentation.")

        seg_logits = self.memory_video_seg_head(feat_dict)

        return {
            "seg_logits": seg_logits,
            "cls_logits": None,
            "features": feat_dict,
        }

    # ==========================================================
    # video classification
    # ==========================================================
    def forward_video_cls(
        self,
        feat_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        """
        if self.video_cls_head is None:
            raise RuntimeError("video_cls_head is None, cannot run video classification.")

        if "cls_feat" not in feat_dict:
            raise KeyError("feat_dict missing 'cls_feat' for video classification.")

        cls_feat = feat_dict["cls_feat"]
        cls_logits = self.video_cls_head(cls_feat)  # [B, num_classes]

        return {
            "seg_logits": None,
            "cls_logits": cls_logits,
            "features": feat_dict,
        }

    # ==========================================================
    # route
    # ==========================================================
    def forward(
        self,
        task_type: str,
        feat_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        if task_type == VIDEO_SEG:
            if self.seg_mode == "memory" or self.use_memory_seg:
                return self.forward_video_seg_memory(feat_dict)
            return self.forward_video_seg_simple(feat_dict)

        if task_type == VIDEO_CLS:
            return self.forward_video_cls(feat_dict)

        raise ValueError(f"TemporalRouter only supports video tasks, got task_type={task_type}")