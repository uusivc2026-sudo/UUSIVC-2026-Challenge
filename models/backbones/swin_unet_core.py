# models/backbones/swin_unet_core.py

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.task_defs import IMAGE


class SwinUNetCore(nn.Module):
    """
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
    {
        "encoder_out": Tensor,
        "skip_features": list[Tensor],
        "seg_feat": Tensor,
        "cls_feat": Tensor,
        "meta": {...}
    }

    The organizers keep this baseline note in English for public release.
    - image:
        seg_feat: [B, C_seg, H, W]
        cls_feat: [B, C_cls] or [B, L, C_cls]
    - video:
        seg_feat: [B, T, C_seg, H, W]
        cls_feat: [B, T, C_cls] or [B, T, L, C_cls]
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        seg_feature_channels: int = 96,
        cls_feature_channels: int = 192,
        in_channels: int = 3,
        use_prompt: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.use_prompt = use_prompt
        self.seg_feature_channels = seg_feature_channels
        self.cls_feature_channels = cls_feature_channels

        # ------------------------------------------------------------------
        # The organizers keep this baseline step explicit for participants.
        # The organizers keep this baseline step explicit for participants.
        # ------------------------------------------------------------------
        self.seg_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, seg_feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(seg_feature_channels),
            nn.GELU(),
        )

        self.cls_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, cls_feature_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cls_feature_channels),
            nn.GELU(),
        )

    # ==========================================================
    # helper
    # ==========================================================
    @staticmethod
    def _get_image_tensor(batch: Dict[str, Any]) -> torch.Tensor:
        if IMAGE not in batch:
            raise KeyError(f"Missing required key '{IMAGE}' in batch.")
        return batch[IMAGE]

    def _extract_prompts(
        self,
        batch: Dict[str, Any],
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.use_prompt:
            return None
        if prompts is not None:
            return prompts
        return {
            "position_prompt": batch.get("position_prompt", None),
            "task_prompt": batch.get("task_prompt", None),
            "mode_prompt": batch.get("mode_prompt", None),
            "type_prompt": batch.get("type_prompt", None),
        }

    # ==========================================================
    # encoder hooks
    # ==========================================================
    def encode_image(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        The organizers keep this baseline note in English for public release.
            encoder_out: Tensor
            skip_features: list[Tensor]

        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        """
        if self.encoder is not None and hasattr(self.encoder, "forward_image"):
            return self.encoder.forward_image(x, prompts=prompts)

        # The organizers keep this baseline step explicit for participants.
        # The organizers keep this baseline step explicit for participants.
        feat = self.cls_proj(x)                                   # [B, C_cls, H, W]
        encoder_out = F.adaptive_avg_pool2d(feat, 1).flatten(1)   # [B, C_cls]

        # The organizers keep this baseline step explicit for participants.
        skip1 = self.seg_proj(x)                                  # [B, C_seg, H, W]
        skip2 = F.avg_pool2d(skip1, kernel_size=2, stride=2)      # [B, C_seg, H/2, W/2]
        skip3 = F.avg_pool2d(skip2, kernel_size=2, stride=2)      # [B, C_seg, H/4, W/4]

        return encoder_out, [skip1, skip2, skip3]

    def encode_video(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        """
        x: [B, T, C, H, W]

        The organizers keep this baseline note in English for public release.
            encoder_out: [B, T, C_cls] or richer structure
            skip_features: list of video features
            video_meta: dict
        """
        if self.encoder is not None and hasattr(self.encoder, "forward_video"):
            return self.encoder.forward_video(x, prompts=prompts)

        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)

        feat = self.cls_proj(x_flat)                                  # [B*T, C_cls, H, W]
        encoder_out = F.adaptive_avg_pool2d(feat, 1).flatten(1)       # [B*T, C_cls]
        encoder_out = encoder_out.view(b, t, -1)                      # [B, T, C_cls]

        skip1 = self.seg_proj(x_flat)                                 # [B*T, C_seg, H, W]
        skip2 = F.avg_pool2d(skip1, kernel_size=2, stride=2)
        skip3 = F.avg_pool2d(skip2, kernel_size=2, stride=2)

        skip1 = skip1.view(b, t, *skip1.shape[1:])                    # [B,T,C,H,W]
        skip2 = skip2.view(b, t, *skip2.shape[1:])
        skip3 = skip3.view(b, t, *skip3.shape[1:])

        video_meta = {
            "batch_size": b,
            "num_frames": t,
            "height": h,
            "width": w,
        }
        return encoder_out, [skip1, skip2, skip3], video_meta

    # ==========================================================
    # feature builders
    # ==========================================================
    def build_image_seg_feat(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        """
        seg_feat = skip_features[0]
        return seg_feat

    def build_image_cls_feat(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        """
        cls_map = self.cls_proj(x)
        cls_feat = F.adaptive_avg_pool2d(cls_map, 1).flatten(1)   # [B, C_cls]
        return cls_feat

    def build_video_seg_feat(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        skip_features: List[torch.Tensor],
        video_meta: Dict[str, Any],
    ) -> torch.Tensor:
        """
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
            The organizers keep this baseline note in English for public release.
            2) memory-enhanced video seg
        """
        seg_feat = skip_features[0]
        return seg_feat

    def build_video_cls_feat(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        skip_features: List[torch.Tensor],
        video_meta: Dict[str, Any],
    ) -> torch.Tensor:
        """
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
        """
        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        cls_map = self.cls_proj(x_flat)
        cls_feat = F.adaptive_avg_pool2d(cls_map, 1).flatten(1)   # [B*T, C_cls]
        cls_feat = cls_feat.view(b, t, -1)                        # [B, T, C_cls]
        return cls_feat

    # ==========================================================
    # public API
    # ==========================================================
    def forward_image_features(
        self,
        batch: Dict[str, Any],
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        The organizers keep this baseline note in English for public release.
        """
        x = self._get_image_tensor(batch)
        if x.ndim != 4:
            raise ValueError(f"forward_image_features expects [B,C,H,W], got {x.shape}")

        prompts = self._extract_prompts(batch, prompts=prompts)
        encoder_out, skip_features = self.encode_image(x, prompts=prompts)

        seg_feat = self.build_image_seg_feat(x, encoder_out, skip_features)
        cls_feat = self.build_image_cls_feat(x, encoder_out, skip_features)

        feat_dict = {
            "encoder_out": encoder_out,
            "skip_features": skip_features,
            "seg_feat": seg_feat,
            "cls_feat": cls_feat,
            "meta": {
                "is_video": False,
            },
        }
        return feat_dict

    def forward_video_features(
        self,
        batch: Dict[str, Any],
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        The organizers keep this baseline note in English for public release.
        """
        x = self._get_image_tensor(batch)
        if x.ndim != 5:
            raise ValueError(f"forward_video_features expects [B,T,C,H,W], got {x.shape}")

        prompts = self._extract_prompts(batch, prompts=prompts)
        encoder_out, skip_features, video_meta = self.encode_video(x, prompts=prompts)

        seg_feat = self.build_video_seg_feat(x, encoder_out, skip_features, video_meta)
        cls_feat = self.build_video_cls_feat(x, encoder_out, skip_features, video_meta)

        feat_dict = {
            "encoder_out": encoder_out,
            "skip_features": skip_features,
            "seg_feat": seg_feat,
            "cls_feat": cls_feat,
            "meta": {
                "is_video": True,
                **video_meta,
            },
        }
        return feat_dict