"""Central dataclass config (inspired by HF TrainingArguments)."""

from dataclasses import dataclass, field
from typing import List
import torch


@dataclass
class CFG:
    model_name: str = "bert-base-uncased"
    tasks: List[str] = field(default_factory=lambda: ["sst2", "qnli"])
    train_size: int = 5000
    eval_size: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 4
    batch_train: int = 32
    batch_eval: int = 64
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints"
    wandb_project: str = "direction_aware_fusion"
    wandb_entity: str = "your_wandb_entity"
    wandb_run_name: str = "default_run"
    wandb: bool = True
    log_interval: int = 100

    seeds: List[int] = field(
        default_factory=lambda: [42, 123, 456]
    )
    num_seeds: int = 3
    seed: int = 42
    confidence_interval: float = 0.95
    enable_averaging: bool = (
        False
    )
    save_individual_runs: bool = True
    aggregation_method: str = "mean"
