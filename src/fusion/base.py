"""Base interface for every fusion strategy."""

from abc import ABC, abstractmethod
from typing import List
import torch.nn as nn


class FusionMethod(ABC):
 

    @abstractmethod
    def fuse(self, models: List[nn.Module], **kwargs) -> nn.Module:
        raise NotImplementedError
