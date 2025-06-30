"""Soft (and Greedy) Soup fusion."""

from typing import List
import copy, torch
import torch.nn as nn
from .base import FusionMethod


class SoftSoup(FusionMethod):
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def fuse(self, models: List[nn.Module], **kwargs) -> nn.Module:
        if len(models) != 2:
            raise ValueError("SoftSoup in Test 0 expects exactly two models")
        m1, m2 = models
        fused = copy.deepcopy(m1)
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        blended = {
            k: (1 - self.alpha) * sd1[k] + self.alpha * sd2[k]
            for k in sd1
            if k in sd2 and sd1[k].shape == sd2[k].shape
        }
        fused.load_state_dict(blended, strict=False)
        return fused
