"""Multi-seed experiment utilities for 3-seed averaging and confidence intervals."""

import json
import numpy as np
import os
from typing import Dict, List, Tuple, Any
from scipy import stats
from ..utils.config import CFG
from ..modeling.train import train_task
from ..modeling.evaluate import evaluate
from ..modeling.metrics import cl_metrics


def set_global_seed(seed: int):
    import torch
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_single_seed_experiment(
    fusion_method, task_names: List[str], cfg: CFG, seed: int
) -> Dict[str, Any]:
    cfg.seed = seed
    set_global_seed(seed)

    models = []
    val_datasets = []
    tokenizer = None

    for task in task_names:
        model, val_data, tok = train_task(task, cfg)
        models.append(model)
        val_datasets.append(val_data)
        if tokenizer is None:
            tokenizer = tok

    from transformers import AutoModelForSequenceClassification

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2
    )

    fused_model = fusion_method.fuse(models, base_model=base_model)

    results = {}
    individual_accs = []
    fused_accs = []

    for i, (model, task) in enumerate(zip(models, task_names)):
        task_accs = []
        for j, (val_data, eval_task) in enumerate(zip(val_datasets, task_names)):
            acc = evaluate(model, val_data, tokenizer, cfg)
            task_accs.append(acc)
            results[f"model{i + 1}_{task}_on_{eval_task}"] = acc
        individual_accs.append(task_accs)

    for j, (val_data, eval_task) in enumerate(zip(val_datasets, task_names)):
        acc = evaluate(fused_model, val_data, tokenizer, cfg)
        fused_accs.append(acc)
        results[f"fused_on_{eval_task}"] = acc

    cl_results = {}
    for i, task_accs in enumerate(individual_accs):
        cl_metrics_result = cl_metrics(task_accs, fused_accs, i)
        cl_results[f"model{i + 1}"] = cl_metrics_result

    results["cl_metrics"] = cl_results
    results["individual_accuracies"] = individual_accs
    results["fused_accuracies"] = fused_accs
    results["seed"] = seed

    return results


def run_multi_seed_experiment(
    fusion_method, task_names: List[str], cfg: CFG, experiment_name: str
) -> Dict[str, Any]:
    if not cfg.enable_averaging:
        return run_single_seed_experiment(fusion_method, task_names, cfg, cfg.seed)

    all_results = []

    for seed in cfg.seeds[: cfg.num_seeds]:
        print(f"Running experiment with seed {seed}...")
        seed_results = run_single_seed_experiment(fusion_method, task_names, cfg, seed)
        all_results.append(seed_results)

        if cfg.save_individual_runs:
            os.makedirs("logs/individual_runs", exist_ok=True)
            with open(
                f"logs/individual_runs/{experiment_name}_seed_{seed}.json", "w"
            ) as f:
                json.dump(seed_results, f, indent=2)

    aggregated_results = aggregate_multi_seed_results(all_results, cfg)
    aggregated_results["experiment_name"] = experiment_name
    aggregated_results["seeds_used"] = cfg.seeds[: cfg.num_seeds]
    aggregated_results["num_seeds"] = cfg.num_seeds

    return aggregated_results


def aggregate_multi_seed_results(all_results: List[Dict], cfg: CFG) -> Dict[str, Any]:
    aggregated = {}

    sample_result = all_results[0]
    metric_keys = [
        k
        for k in sample_result.keys()
        if k not in ["seed", "cl_metrics", "individual_accuracies", "fused_accuracies"]
    ]

    for key in metric_keys:
        values = [result[key] for result in all_results if key in result]
        if values and isinstance(values[0], (int, float)):
            mean_val, ci_low, ci_high, std_val = compute_confidence_interval(
                values, cfg.confidence_interval
            )
            aggregated[key] = {
                "mean": mean_val,
                "std": std_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "values": values,
            }

    cl_metrics_aggregated = {}
    for model_key in sample_result["cl_metrics"].keys():
        cl_metrics_aggregated[model_key] = {}
        for metric_name in sample_result["cl_metrics"][model_key].keys():
            values = [
                result["cl_metrics"][model_key][metric_name] for result in all_results
            ]
            mean_val, ci_low, ci_high, std_val = compute_confidence_interval(
                values, cfg.confidence_interval
            )
            cl_metrics_aggregated[model_key][metric_name] = {
                "mean": mean_val,
                "std": std_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "values": values,
            }

    aggregated["cl_metrics"] = cl_metrics_aggregated

    individual_accs = np.array(
        [result["individual_accuracies"] for result in all_results]
    )
    fused_accs = np.array([result["fused_accuracies"] for result in all_results])

    aggregated["individual_accuracies"] = {
        "mean": np.mean(individual_accs, axis=0).tolist(),
        "std": np.std(individual_accs, axis=0).tolist(),
        "values": individual_accs.tolist(),
    }

    aggregated["fused_accuracies"] = {
        "mean": np.mean(fused_accs, axis=0).tolist(),
        "std": np.std(fused_accs, axis=0).tolist(),
        "values": fused_accs.tolist(),
    }

    return aggregated


def compute_confidence_interval(
    values: List[float], confidence: float = 0.95
) -> Tuple[float, float, float, float]:
    """Compute mean, confidence interval bounds, and standard deviation."""
    values = np.array(values)
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0

    if len(values) <= 1:
        return mean_val, mean_val, mean_val, std_val

    alpha = 1 - confidence
    dof = len(values) - 1
    t_critical = stats.t.ppf(1 - alpha / 2, dof)
    margin_error = t_critical * (std_val / np.sqrt(len(values)))

    ci_low = mean_val - margin_error
    ci_high = mean_val + margin_error

    return mean_val, ci_low, ci_high, std_val


def save_aggregated_results(results: Dict[str, Any], experiment_name: str):
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/{experiment_name}_aggregated_metrics.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Aggregated results saved to {filename}")
