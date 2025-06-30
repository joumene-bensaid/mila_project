from transformers import AutoTokenizer
from .config import CFG


def get_tokenizer(cfg: CFG):
    return AutoTokenizer.from_pretrained(cfg.model_name)
