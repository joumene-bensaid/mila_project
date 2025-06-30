import argparse, os, json
from transformers import AutoModelForSequenceClassification
from src.utils.config import CFG
from src.modeling.train import train_task
from src.modeling.evaluate import evaluate
from src.modeling.metrics import cl_metrics
from src.fusion.orthogonal import OrthogonalDeltas
from src.utils.wandb_utils import init_wandb
from src.modeling.delta import normalize

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true")
args = parser.parse_args()

cfg = CFG()
run = init_wandb(cfg, "test1b_orthogonal_normalized") if args.wandb else None

m1, val1, tok = train_task(cfg.tasks[0], cfg)
m2, val2, _ = train_task(cfg.tasks[1], cfg)
base = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=2)

fuser = OrthogonalDeltas()
fused = fuser.fuse([m1, m2], base_model=base)

acc1 = [evaluate(m1, val1, tok, cfg), evaluate(m1, val2, tok, cfg)]
acc2 = [evaluate(m2, val1, tok, cfg), evaluate(m2, val2, tok, cfg)]
accF = [evaluate(fused, val1, tok, cfg), evaluate(fused, val2, tok, cfg)]

cl_metrics_1 = cl_metrics(acc1, accF, 0)
cl_metrics_2 = cl_metrics(acc2, accF, 1)

print("Orthogonal+Norm accuracies", acc1, acc2, accF)
print("CL", cl_metrics_1, cl_metrics_2)

os.makedirs("logs", exist_ok=True)
with open("logs/test1b_metrics.json", "w") as f:
    json.dump({"model1": cl_metrics_1, "model2": cl_metrics_2}, f, indent=2)

if run:
    import wandb

    run.log(
        {
            f"model1_{cfg.tasks[0]}_on_{cfg.tasks[0]}": acc1[0],
            f"model1_{cfg.tasks[0]}_on_{cfg.tasks[1]}": acc1[1],
            f"model2_{cfg.tasks[1]}_on_{cfg.tasks[0]}": acc2[0],
            f"model2_{cfg.tasks[1]}_on_{cfg.tasks[1]}": acc2[1],
            f"fused_on_{cfg.tasks[0]}": accF[0],
            f"fused_on_{cfg.tasks[1]}": accF[1],
            "cl_metrics_task0_retention": cl_metrics_1["Retention%"],
            "cl_metrics_task0_transfer": cl_metrics_1["Transfer%"],
            "cl_metrics_task1_retention": cl_metrics_2["Retention%"],
            "cl_metrics_task1_transfer": cl_metrics_2["Transfer%"],
            "avg_individual_accuracy": (acc1[0] + acc2[1]) / 2,
            "avg_fused_accuracy": (accF[0] + accF[1]) / 2,
            "forgetting_task0": max(0, acc1[0] - accF[0]),
            "forgetting_task1": max(0, acc2[1] - accF[1]),
        }
    )
    run.finish()
