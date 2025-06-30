from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from ..utils.config import CFG
from sklearn.metrics import accuracy_score
import numpy as np


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": accuracy_score(labels, np.argmax(logits, 1))}


def evaluate(model, dataset, tokenizer, cfg: CFG):
    trainer = Trainer(
        model=model.to(cfg.device),
        args=TrainingArguments(
            output_dir="./tmp_eval",
            report_to="none",
            per_device_eval_batch_size=cfg.batch_eval,
        ),
        eval_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=_compute_metrics,
    )
    res = trainer.evaluate()
    model.cpu()
    return res["eval_accuracy"]
