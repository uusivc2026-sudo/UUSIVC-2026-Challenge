# models/heads/seg_head.py

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: bool = False,
        act_layer: nn.Module = nn.GELU,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SegHead2D(nn.Module):
    """
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
        x: [B, C, H, W]
    The organizers keep this baseline note in English for public release.
        logits: [B, num_classes, H, W]

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        upsample_scale: int = 1,
    ):
        super().__init__()
        self.upsample_scale = upsample_scale

        self.block = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=3),
            nn.Dropout2d(dropout),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3),
        )
        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"SegHead2D expects [B,C,H,W], got {x.shape}")

        x = self.block(x)
        logits = self.classifier(x)

        if self.upsample_scale != 1:
            logits = F.interpolate(
                logits,
                scale_factor=self.upsample_scale,
                mode="bilinear",
                align_corners=False,
            )

        return logits


class SegHeadVideoSimple(nn.Module):
    """
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
        x: [B, T, C, H, W]
    The organizers keep this baseline note in English for public release.
        logits: [B, T, num_classes, H, W]

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
    - baseline video segmentation
    The organizers keep this baseline note in English for public release.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        upsample_scale: int = 1,
    ):
        super().__init__()
        self.frame_head = SegHead2D(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_classes=num_classes,
            dropout=dropout,
            upsample_scale=upsample_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"SegHeadVideoSimple expects [B,T,C,H,W], got {x.shape}")

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        logits = self.frame_head(x)   # [B*T, K, H', W']
        logits = logits.reshape(b, t, *logits.shape[1:])
        return logits