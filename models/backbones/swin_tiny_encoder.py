from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_T_Weights, swin_t


class SwinTinyEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        pretrained: bool = False,
        pretrained_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"SwinTinyEncoder currently supports in_channels=3, got {in_channels}")

        weights = Swin_T_Weights.DEFAULT if pretrained and not pretrained_checkpoint else None
        self.backbone = swin_t(weights=weights)
        self.img_size = img_size
        self.stage_channels = [96, 192, 384, 768]

        if pretrained_checkpoint:
            self.load_swin_checkpoint(pretrained_checkpoint)

    @staticmethod
    def _map_swin_unet_key(key: str) -> Optional[str]:
        if key.startswith("patch_embed.proj."):
            return key.replace("patch_embed.proj.", "features.0.0.")
        if key.startswith("patch_embed.norm."):
            return key.replace("patch_embed.norm.", "features.0.2.")

        for layer_idx, feature_idx in enumerate([1, 3, 5, 7]):
            block_prefix = f"layers.{layer_idx}.blocks."
            if key.startswith(block_prefix):
                rest = key[len(block_prefix):]
                if rest.endswith("attn_mask") or "relative_position_index" in rest:
                    return None
                rest = rest.replace("mlp.fc1.", "mlp.0.").replace("mlp.fc2.", "mlp.3.")
                return f"features.{feature_idx}.{rest}"

            downsample_prefix = f"layers.{layer_idx}.downsample."
            if key.startswith(downsample_prefix):
                rest = key[len(downsample_prefix):]
                return f"features.{feature_idx + 1}.{rest}"

        if key.startswith("norm.") or key.startswith("head."):
            return key
        return key

    def load_swin_checkpoint(self, checkpoint_path: str) -> None:
        resolved_path = Path(checkpoint_path).expanduser().resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"Swin pretrained checkpoint not found: {resolved_path}")

        checkpoint = torch.load(str(resolved_path), map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        own_state = self.backbone.state_dict()
        mapped_state = {}
        skipped = []

        for key, value in state_dict.items():
            mapped_key = self._map_swin_unet_key(key)
            if mapped_key is None:
                skipped.append(key)
                continue
            if mapped_key in own_state and own_state[mapped_key].shape == value.shape:
                mapped_state[mapped_key] = value
            else:
                skipped.append(key)

        missing, unexpected = self.backbone.load_state_dict(mapped_state, strict=False)
        print(
            "[Info] Loaded local Swin checkpoint: "
            f"{resolved_path} matched={len(mapped_state)} "
            f"missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}"
        )

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def _forward_single_image(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        if x.ndim != 4:
            raise ValueError(f"SwinTinyEncoder expects [B,C,H,W], got {tuple(x.shape)}")

        if x.shape[-2] != self.img_size or x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)

        feats = self.backbone.features
        x0 = feats[0](x)
        s1 = feats[1](x0)
        x2 = feats[2](s1)
        s2 = feats[3](x2)
        x4 = feats[4](s2)
        s3 = feats[5](x4)
        x6 = feats[6](s3)
        s4 = feats[7](x6)

        encoder_tokens = self.backbone.norm(s4)
        encoder_out = encoder_tokens.mean(dim=(1, 2))

        return encoder_out, [
            self._to_nchw(s1),
            self._to_nchw(s2),
            self._to_nchw(s3),
            self._to_nchw(s4),
        ], {
            "image_size": self.img_size,
            "stage_channels": self.stage_channels,
        }

    def forward_image(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        encoder_out, skip_features, _ = self._forward_single_image(x)
        return encoder_out, skip_features

    def forward_video(
        self,
        x: torch.Tensor,
        prompts: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict[str, Any]]:
        if x.ndim != 5:
            raise ValueError(f"SwinTinyEncoder expects [B,T,C,H,W], got {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        x_flat = x.reshape(b * t, c, h, w)
        encoder_out, skip_features, meta = self._forward_single_image(x_flat)
        encoder_out = encoder_out.view(b, t, -1)
        video_skips = [feat.view(b, t, *feat.shape[1:]) for feat in skip_features]
        meta.update(
            {
                "batch_size": b,
                "num_frames": t,
                "height": h,
                "width": w,
            }
        )
        return encoder_out, video_skips, meta
