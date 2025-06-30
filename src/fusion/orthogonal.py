"""Orthogonal delta fusion (Test 1)."""

from typing import List
import torch, torch.nn as nn, copy
from .base import FusionMethod
from ..modeling.delta import learning_vector, project_orthogonal, vector_to_model


class OrthogonalDeltas(FusionMethod):
    

    def fuse(
        self, models: List[nn.Module], base_model: nn.Module, **kwargs
    ) -> nn.Module:
        if len(models) != 2:
            raise ValueError("Orthogonal fusion expects exactly two models")
        m1, m2 = models
        vec1 = learning_vector(m1, base_model)
        vec2 = learning_vector(m2, base_model)
        vec2_orth = project_orthogonal(vec2, vec1)
        fused_vec = vec1 + vec2_orth
        return vector_to_model(fused_vec, base_model)
