import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAwareNeighborAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd number, got {kernel_size}")

        self.dim = dim
        self.num_classes = num_classes
        self.kernel_size = kernel_size

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.cls_proj = nn.Linear(dim, num_classes)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"ClassAwareNeighborAttention expects [B,T,C], got {tuple(x.shape)}")

        residual = x
        x = self.norm(x)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        cls_query = self.cls_proj(x)

        pad = self.kernel_size // 2
        k = F.pad(k.transpose(1, 2), (pad, pad), mode="replicate").transpose(1, 2)
        v = F.pad(v.transpose(1, 2), (pad, pad), mode="replicate").transpose(1, 2)
        cls_key = F.pad(cls_query.transpose(1, 2), (pad, pad), mode="replicate").transpose(1, 2)

        k = k.unfold(dimension=1, size=self.kernel_size, step=1).permute(0, 1, 3, 2)
        v = v.unfold(dimension=1, size=self.kernel_size, step=1).permute(0, 1, 3, 2)
        cls_key = cls_key.unfold(dimension=1, size=self.kernel_size, step=1).permute(0, 1, 3, 2)

        feature_scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(self.dim)
        class_scores = (cls_query.unsqueeze(2) * cls_key).sum(dim=-1) / math.sqrt(self.num_classes)
        attn = torch.softmax(feature_scores + class_scores, dim=-1)

        context = (attn.unsqueeze(-1) * v).sum(dim=2)
        context = self.out_proj(context)
        return residual + context
