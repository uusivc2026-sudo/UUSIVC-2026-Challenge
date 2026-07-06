import torch
import torch.nn as nn


class AttentionPool1d(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.score = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"AttentionPool1d expects [B,L,C], got {tuple(x.shape)}")
        weights = torch.softmax(self.score(self.norm(x)), dim=1)
        return (weights * x).sum(dim=1)


class ResidualMLPBlock(nn.Module):
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


class ClsHead(nn.Module):
    """
    Classification head for image tasks.

    Supports:
    - pooled features: [B, C]
    - token features: [B, L, C]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 0,
        num_classes: int = 2,
        dropout: float = 0.1,
        mode: str = "mlp",
        num_layers: int = 2,
    ):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        hidden = hidden_channels if hidden_channels and hidden_channels > 0 else in_channels * 2
        self.dropout = nn.Dropout(dropout)
        self.attn_pool = AttentionPool1d(in_channels)

        if mode == "perceptguide":
            fused_channels = in_channels * 3
            self.fuse = nn.Sequential(
                nn.LayerNorm(fused_channels),
                nn.Linear(fused_channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, in_channels),
            )
            depth = max(int(num_layers), 1)
            self.res_blocks = nn.Sequential(
                *[
                    ResidualMLPBlock(
                        channels=in_channels,
                        hidden_channels=hidden,
                        dropout=dropout,
                    )
                    for _ in range(depth)
                ]
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
            return

        if hidden_channels and hidden_channels > 0:
            self.classifier = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_channels, num_classes)

    def _pool_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.mean(dim=1)
        raise ValueError(f"ClsHead expects [B,C] or [B,L,C], got {tuple(x.shape)}")

    def _perceptguide_pool(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            tokens = x.unsqueeze(1)
        elif x.ndim == 3:
            tokens = x
        else:
            raise ValueError(f"ClsHead expects [B,C] or [B,L,C], got {tuple(x.shape)}")

        mean_pool = tokens.mean(dim=1)
        max_pool = tokens.max(dim=1).values
        attn_pool = self.attn_pool(tokens)
        fused = torch.cat([mean_pool, max_pool, attn_pool], dim=-1)
        fused = self.fuse(fused)
        return self.res_blocks(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "perceptguide":
            x = self._perceptguide_pool(x)
        else:
            x = self._pool_tokens(x)
        x = self.dropout(x)
        return self.classifier(x)
