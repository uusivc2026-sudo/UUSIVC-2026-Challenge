# models/heads/temporal_cls_head.py

from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.video.neighbor_attention import ClassAwareNeighborAttention


class TemporalAttentionPool(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.score = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TemporalAttentionPool expects [B,T,C], got {tuple(x.shape)}")
        weights = torch.softmax(self.score(self.norm(x)), dim=1)
        return (weights * x).sum(dim=1)


class TemporalResidualMLPBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class TemporalPoolClassifier(nn.Module):
    """
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
        logits: [B, num_classes]

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        dropout: float = 0.1,
        temporal_pool: Literal["mean", "max"] = "mean",
        hidden_channels: int = 0,
    ):
        super().__init__()
        self.temporal_pool = temporal_pool
        self.dropout = nn.Dropout(dropout)

        if hidden_channels and hidden_channels > 0:
            self.cls = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes),
            )
        else:
            self.cls = nn.Linear(in_channels, num_classes)

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
            [B,T,C] or [B,T,L,C]
        """
        if x.ndim == 3:
            return x
        if x.ndim == 4:
            return x.mean(dim=2)
        raise ValueError(f"TemporalPoolClassifier expects [B,T,C] or [B,T,L,C], got {x.shape}")

    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,C]
        """
        if self.temporal_pool == "mean":
            return x.mean(dim=1)
        if self.temporal_pool == "max":
            return x.max(dim=1).values
        raise ValueError(f"Unsupported temporal_pool={self.temporal_pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pool_tokens(x)      # [B,T,C]
        x = self._pool_time(x)        # [B,C]
        x = self.dropout(x)
        logits = self.cls(x)
        return logits


class NeighborAttentionClassifier(nn.Module):
    """
    The organizers keep this baseline note in English for public release.

    The organizers keep this baseline note in English for public release.
        The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
        logits: [B, num_classes]

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
        in_channels: int,
        num_classes: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 3,
        temporal_pool: Literal["mean", "max"] = "mean",
        hidden_channels: int = 0,
    ):
        super().__init__()
        self.temporal_pool = temporal_pool
        self.neighbor_attn = ClassAwareNeighborAttention(
            dim=in_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
        )
        self.dropout = nn.Dropout(dropout)

        if hidden_channels and hidden_channels > 0:
            self.cls = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes),
            )
        else:
            self.cls = nn.Linear(in_channels, num_classes)

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x
        if x.ndim == 4:
            return x.mean(dim=2)
        raise ValueError(f"NeighborAttentionClassifier expects [B,T,C] or [B,T,L,C], got {x.shape}")

    def _pool_time(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_pool == "mean":
            return x.mean(dim=1)
        if self.temporal_pool == "max":
            return x.max(dim=1).values
        raise ValueError(f"Unsupported temporal_pool={self.temporal_pool}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pool_tokens(x)      # [B,T,C]
        x = self.neighbor_attn(x)     # [B,T,C]
        x = self._pool_time(x)        # [B,C]
        x = self.dropout(x)
        logits = self.cls(x)
        return logits


class PerceptGuideTemporalClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        dropout: float = 0.1,
        kernel_size: int = 3,
        hidden_channels: int = 0,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        mlp_layers: int = 2,
    ):
        super().__init__()
        hidden = hidden_channels if hidden_channels and hidden_channels > 0 else in_channels * 2
        self.token_attn_pool = TemporalAttentionPool(in_channels)
        self.neighbor_attn = ClassAwareNeighborAttention(
            dim=in_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
        )
        if transformer_layers and transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_channels,
                nhead=max(int(transformer_heads), 1),
                dim_feedforward=hidden,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=int(transformer_layers),
            )
        else:
            self.temporal_encoder = nn.Identity()
        self.temporal_attn_pool = TemporalAttentionPool(in_channels)
        fused_channels = in_channels * 3
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_channels),
            nn.Linear(fused_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_channels),
        )
        self.res_blocks = nn.Sequential(
            *[
                TemporalResidualMLPBlock(
                    channels=in_channels,
                    hidden_channels=hidden,
                    dropout=dropout,
                )
                for _ in range(max(int(mlp_layers), 1))
            ]
        )
        self.cls = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return x
        if x.ndim == 4:
            b, t, l, c = x.shape
            tokens = x.reshape(b * t, l, c)
            mean_pool = tokens.mean(dim=1)
            max_pool = tokens.max(dim=1).values
            attn_pool = self.token_attn_pool(tokens)
            return ((mean_pool + max_pool + attn_pool) / 3.0).view(b, t, c)
        raise ValueError(f"PerceptGuideTemporalClassifier expects [B,T,C] or [B,T,L,C], got {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pool_tokens(x)
        x = self.neighbor_attn(x)
        x = self.temporal_encoder(x)
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values
        attn_pool = self.temporal_attn_pool(x)
        x = self.fuse(torch.cat([mean_pool, max_pool, attn_pool], dim=-1))
        x = self.res_blocks(x)
        return self.cls(x)


class TemporalClsHead(nn.Module):
    """
    The organizers keep this baseline note in English for public release.

    mode:
        - "mean": TemporalPoolClassifier(mean)
        - "max": TemporalPoolClassifier(max)
        - "neighbor": NeighborAttentionClassifier
        - "perceptguide": neighbor attention + temporal transformer + multi-pool MLP

    The organizers keep this baseline note in English for public release.
    The organizers keep this baseline note in English for public release.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        mode: Literal["mean", "max", "neighbor", "perceptguide"] = "mean",
        dropout: float = 0.1,
        hidden_channels: int = 0,
        kernel_size: int = 3,
        transformer_layers: int = 1,
        transformer_heads: int = 4,
        mlp_layers: int = 2,
    ):
        super().__init__()
        self.mode = mode

        if mode == "mean":
            self.impl = TemporalPoolClassifier(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
                temporal_pool="mean",
                hidden_channels=hidden_channels,
            )
        elif mode == "max":
            self.impl = TemporalPoolClassifier(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
                temporal_pool="max",
                hidden_channels=hidden_channels,
            )
        elif mode == "neighbor":
            self.impl = NeighborAttentionClassifier(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
                kernel_size=kernel_size,
                temporal_pool="mean",
                hidden_channels=hidden_channels,
            )
        elif mode == "perceptguide":
            self.impl = PerceptGuideTemporalClassifier(
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
                kernel_size=kernel_size,
                hidden_channels=hidden_channels,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                mlp_layers=mlp_layers,
            )
        else:
            raise ValueError(f"Unsupported TemporalClsHead mode={mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)
