"""Shared training routine."""

import logging
import traceback
import sys
from typing import Tuple
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from ..utils.config import CFG
from ..utils.tokenizer import get_tokenizer
from sklearn.metrics import accuracy_score
import torch, numpy as np

__all__ = ["train_task"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_debug.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


def _preprocess(ex, task, tok):
    if task == "sst2":
        return tok(ex["sentence"], truncation=True)
    return tok(ex["question"], ex["sentence"], truncation=True)


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": accuracy_score(labels, np.argmax(logits, 1))}


def train_task(task: str, cfg: CFG):
    """Train a model on a specific task with comprehensive error handling."""
    try:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.cuda.manual_seed_all(cfg.seed)

        logging.info(f"Starting training for task: {task} with seed: {cfg.seed}")
        logging.info(f"Config: {cfg}")

        if torch.cuda.is_available():
            logging.info(f"CUDA available: {torch.cuda.get_device_name()}")
        else:
            logging.warning("CUDA not available, using CPU")

        tok = get_tokenizer(cfg)
        logging.info("Tokenizer loaded successfully")

        logging.info(f"Loading dataset: glue/{task}")
        ds = load_dataset("glue", task)
        logging.info(
            f"Dataset loaded. Train size: {len(ds['train'])}, Val size: {len(ds['validation'])}"
        )

        tr = (
            ds["train"]
            .shuffle(seed=cfg.seed)
            .select(range(min(cfg.train_size, len(ds["train"]))))
            .map(lambda x: _preprocess(x, task, tok), batched=True)
        )
        val = (
            ds["validation"]
            .shuffle(seed=cfg.seed)
            .select(range(min(cfg.eval_size, len(ds["validation"]))))
            .map(lambda x: _preprocess(x, task, tok), batched=True)
        )
        logging.info(
            f"Data preprocessing complete. Train samples: {len(tr)}, Val samples: {len(val)}"
        )

        logging.info(f"Loading model: {cfg.model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=2
        )
        model.to(cfg.device)
        logging.info(
            f"Model loaded and moved to device: {cfg.device}"
        )
        logging.info("Setting up training arguments")
        args = TrainingArguments(
            output_dir=f"artifacts/{task}",
            eval_strategy="epoch",
            save_strategy="no",
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_train,
            per_device_eval_batch_size=cfg.batch_eval,
            report_to="none",
            logging_steps=10,
            logging_dir=f"artifacts/{task}/logs",
        )

        logging.info("Setting up trainer")
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tr,
            eval_dataset=val,
            tokenizer=tok,
            data_collator=DataCollatorWithPadding(tok),
            compute_metrics=_compute_metrics,
        )

        logging.info("Starting training...")
        trainer.train()
        logging.info("Training completed successfully")

        model.cpu()
        logging.info(f"Model moved to CPU and training finished for task: {task}")
        return model, val, tok

    except Exception as e:
        logging.error(f"ERROR in train_task for {task}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error(f"Error message: {str(e)}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())

        print(f"CRITICAL ERROR in train_task for {task}:", file=sys.stderr)
        print(f"Error: {str(e)}", file=sys.stderr)
        print("Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

        raise
