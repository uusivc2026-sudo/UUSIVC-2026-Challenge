from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-5, ignore_index: Optional[int] = None):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        softmax: bool = True,
    ) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"DiceLoss expects logits [B,C,H,W], got {tuple(logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"DiceLoss expects target [B,H,W], got {tuple(target.shape)}")

        probs = torch.softmax(logits, dim=1) if softmax else logits
        num_classes = probs.shape[1]
        target = target.long()

        valid_mask = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target.masked_fill(~valid_mask, 0)

        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        valid_mask = valid_mask.unsqueeze(1)

        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        reduce_dims = (0, 2, 3)
        intersection = (probs * target_one_hot).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()
