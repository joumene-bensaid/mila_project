import argparse, os, json
from transformers import AutoModelForSequenceClassification
from src.utils.config import CFG
from src.utils.multi_seed import run_multi_seed_experiment, save_aggregated_results
from src.fusion.soft import SoftSoup
from src.utils.wandb_utils import init_wandb

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--multi-seed", action="store_true", help="Run with multiple seeds")
parser.add_argument(
    "--seeds", nargs="+", type=int, default=[42, 123, 456], help="Seeds to use"
)
args = parser.parse_args()

cfg = CFG()

if args.multi_seed:
    cfg.enable_averaging = True
    cfg.seeds = args.seeds
    cfg.num_seeds = len(args.seeds)
    cfg.save_individual_runs = True

run = (
    init_wandb(
        cfg, "test0_soft_soup_multi_seed" if args.multi_seed else "test0_soft_soup"
    )
    if args.wandb
    else None
)

fuser = SoftSoup()

if cfg.enable_averaging:
    results = run_multi_seed_experiment(fuser, cfg.tasks, cfg, "test0")

    save_aggregated_results(results, "test0")

    if run:
        import wandb

        cl_metrics_model1 = results["cl_metrics"]["model1"]
        cl_metrics_model2 = results["cl_metrics"]["model2"]

        run.log(
            {
                "avg_fused_accuracy_mean": sum(results["fused_accuracies"]["mean"])
                / len(results["fused_accuracies"]["mean"]),
                "avg_individual_accuracy_mean": sum(
                    [sum(acc) for acc in results["individual_accuracies"]["mean"]]
                )
                / (
                    len(results["individual_accuracies"]["mean"])
                    * len(results["individual_accuracies"]["mean"][0])
                ),
                "cl_metrics_task0_retention_mean": cl_metrics_model1["Retention%"][
                    "mean"
                ],
                "cl_metrics_task0_retention_ci_width": cl_metrics_model1["Retention%"][
                    "ci_high"
                ]
                - cl_metrics_model1["Retention%"]["ci_low"],
                "cl_metrics_task1_retention_mean": cl_metrics_model2["Retention%"][
                    "mean"
                ],
                "cl_metrics_task1_retention_ci_width": cl_metrics_model2["Retention%"][
                    "ci_high"
                ]
                - cl_metrics_model2["Retention%"]["ci_low"],
                "forgetting_task0_mean": cl_metrics_model1["BWT"]["mean"],
                "forgetting_task1_mean": cl_metrics_model2["BWT"]["mean"],
                "num_seeds": cfg.num_seeds,
                "seeds_used": cfg.seeds[: cfg.num_seeds],
            }
        )

    print(f"✓ Multi-seed experiment completed with {cfg.num_seeds} seeds")
    print(f"Results saved to logs/test0_aggregated_metrics.json")

else:
    from src.modeling.train import train_task
    from src.modeling.evaluate import evaluate
    from src.modeling.metrics import cl_metrics

    m1, val1, tok = train_task(cfg.tasks[0], cfg)
    m2, val2, _ = train_task(cfg.tasks[1], cfg)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2
    )

    fused = fuser.fuse([m1, m2], base_model=base)

    acc1 = [evaluate(m1, val1, tok, cfg), evaluate(m1, val2, tok, cfg)]
    acc2 = [evaluate(m2, val1, tok, cfg), evaluate(m2, val2, tok, cfg)]
    accF = [evaluate(fused, val1, tok, cfg), evaluate(fused, val2, tok, cfg)]

    cl_metrics_1 = cl_metrics(acc1, accF, 0)
    cl_metrics_2 = cl_metrics(acc2, accF, 1)

    print("SoftSoup accuracies", acc1, acc2, accF)
    print("CL", cl_metrics_1, cl_metrics_2)

    os.makedirs("logs", exist_ok=True)
    with open("logs/test0_metrics.json", "w") as f:
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
                "avg_fused_accuracy": sum(accF) / len(accF),
                "avg_individual_accuracy": sum([sum(acc) for acc in [acc1, acc2]]) / 4,
                "cl_metrics_task0_retention": cl_metrics_1["Retention%"],
                "cl_metrics_task0_transfer": cl_metrics_1["Transfer%"],
                "cl_metrics_task1_retention": cl_metrics_2["Retention%"],
                "cl_metrics_task1_transfer": cl_metrics_2["Transfer%"],
                "forgetting_task0": cl_metrics_1["BWT"],
                "forgetting_task1": cl_metrics_2["BWT"],
            }
        )

if run:
    run.finish()
