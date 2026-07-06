from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.swin_tiny_encoder import SwinTinyEncoder
from models.task_defs import IMAGE


class SwinUNetBackbone(nn.Module):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        seg_feature_channels: int = 96,
        cls_feature_channels: int = 192,
        in_channels: int = 3,
        use_prompt: bool = False,
        img_size: int = 224,
        pretrained: bool = False,
    ):
        super().__init__()
        self.encoder = encoder or SwinTinyEncoder(
            img_size=img_size,
            in_channels=in_channels,
            pretrained=pretrained,
        )
        self.use_prompt = use_prompt

        encoder_stage_channels = getattr(self.encoder, "stage_channels", [96, 192, 384, 768])
        self.seg_laterals = nn.ModuleList(
            [
                nn.Conv2d(ch, seg_feature_channels, kernel_size=1, bias=False)
                for ch in encoder_stage_channels
            ]
        )
        self.seg_smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(seg_feature_channels, seg_feature_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(seg_feature_channels),
                    nn.GELU(),
                )
                for _ in encoder_stage_channels
            ]
        )
        self.cls_proj = nn.Sequential(
            nn.Linear(encoder_stage_channels[-1], cls_feature_channels),
            nn.LayerNorm(cls_feature_channels),
            nn.GELU(),
        )
        self.cls_skip_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(ch, cls_feature_channels),
                    nn.LayerNorm(cls_feature_channels),
                    nn.GELU(),
                )
                for ch in encoder_stage_channels
            ]
        )
        self.cls_multiscale_fuse = nn.Sequential(
            nn.LayerNorm(cls_feature_channels * (len(encoder_stage_channels) + 1)),
            nn.Linear(cls_feature_channels * (len(encoder_stage_channels) + 1), cls_feature_channels),
            nn.GELU(),
        )

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

    def encode_image(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        return self.encoder.forward_image(x, prompts=prompts)

    def encode_video(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        return self.encoder.forward_video(x, prompts=prompts)

    def build_image_seg_feat(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        laterals = [proj(feat) for proj, feat in zip(self.seg_laterals, skip_features)]
        fused = self.seg_smooth[-1](laterals[-1])
        for idx in range(len(laterals) - 2, -1, -1):
            fused = F.interpolate(
                fused,
                size=laterals[idx].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = self.seg_smooth[idx](laterals[idx] + fused)
        return fused

    def build_image_cls_feat(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        skip_features: List[torch.Tensor],
    ) -> torch.Tensor:
        pooled_features = [self.cls_proj(encoder_out)]
        for proj, feat in zip(self.cls_skip_projs, skip_features):
            pooled = feat.mean(dim=(-2, -1))
            pooled_features.append(proj(pooled))
        return self.cls_multiscale_fuse(torch.cat(pooled_features, dim=-1))

    def build_video_seg_feat(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        skip_features: List[torch.Tensor],
        video_meta: Dict[str, Any],
    ) -> torch.Tensor:
        _, t = x.shape[:2]
        fused_frames = []
        for frame_idx in range(t):
            frame_skips = [feat[:, frame_idx] for feat in skip_features]
            fused_frames.append(self.build_image_seg_feat(x[:, frame_idx], encoder_out[:, frame_idx], frame_skips))
        return torch.stack(fused_frames, dim=1)

    def build_video_cls_feat(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        skip_features: List[torch.Tensor],
        video_meta: Dict[str, Any],
    ) -> torch.Tensor:
        b, t, c = encoder_out.shape
        pooled_features = [self.cls_proj(encoder_out.reshape(b * t, c))]
        for proj, feat in zip(self.cls_skip_projs, skip_features):
            if feat.ndim != 5:
                raise ValueError(f"Video skip feature must be [B,T,C,H,W], got {tuple(feat.shape)}")
            pooled = feat.mean(dim=(-2, -1)).reshape(b * t, feat.shape[2])
            pooled_features.append(proj(pooled))
        cls_feat = self.cls_multiscale_fuse(torch.cat(pooled_features, dim=-1))
        return cls_feat.view(b, t, -1)

    def forward_image_features(
        self,
        batch: Dict[str, Any],
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        x = self._get_image_tensor(batch)
        if x.ndim != 4:
            raise ValueError(f"forward_image_features expects [B,C,H,W], got {x.shape}")

        prompts = self._extract_prompts(batch, prompts=prompts)
        encoder_out, skip_features = self.encode_image(x, prompts=prompts)
        seg_feat = self.build_image_seg_feat(x, encoder_out, skip_features)
        cls_feat = self.build_image_cls_feat(x, encoder_out, skip_features)

        return {
            "encoder_out": encoder_out,
            "skip_features": skip_features,
            "seg_feat": seg_feat,
            "cls_feat": cls_feat,
            "meta": {"is_video": False},
        }

    def forward_video_features(
        self,
        batch: Dict[str, Any],
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        x = self._get_image_tensor(batch)
        if x.ndim != 5:
            raise ValueError(f"forward_video_features expects [B,T,C,H,W], got {x.shape}")

        prompts = self._extract_prompts(batch, prompts=prompts)
        encoder_out, skip_features, video_meta = self.encode_video(x, prompts=prompts)
        seg_feat = self.build_video_seg_feat(x, encoder_out, skip_features, video_meta)
        cls_feat = self.build_video_cls_feat(x, encoder_out, skip_features, video_meta)

        return {
            "encoder_out": encoder_out,
            "skip_features": skip_features,
            "seg_feat": seg_feat,
            "cls_feat": cls_feat,
            "meta": {
                "is_video": True,
                **video_meta,
            },
        }
