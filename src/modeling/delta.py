import torch, copy
import torch.nn.functional as F
from typing import Dict

__all__ = ["learning_vector", "project_orthogonal", "vector_to_model", "normalize"]

def learning_vector(finetuned, base):
    vecs = []
    for n, p in finetuned.named_parameters():
        if p.requires_grad and n in base.state_dict():
            vecs.append((p.data - base.state_dict()[n]).flatten())
    return torch.cat(vecs)

def project_orthogonal(v2, v1):
    proj = (torch.dot(v2, v1) / torch.dot(v1, v1)) * v1
    return v2 - proj

def vector_to_model(delta_vec, base_model):
    new_model = copy.deepcopy(base_model)
    ptr = 0
    for n, p in new_model.named_parameters():
        if p.requires_grad:
            num = p.numel()
            p.data += delta_vec[ptr:ptr+num].view_as(p.data)
            ptr += num
    return new_model

def normalize(vec):
    return vec / vec.norm()