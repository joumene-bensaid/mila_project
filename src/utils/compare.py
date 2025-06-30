import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import List

try:
    import wandb
except ImportError:
    wandb = None

LOG_DIRS = {
    "SoftSoup (Test 0)": "logs/test0_metrics.json",
    "Orthogonal (Test 1)": "logs/test1_metrics.json",
    "Orthogonal + Norm (Test 1B)": "logs/test1b_metrics.json",
}

AGGREGATED_LOG_DIRS = {
    "SoftSoup (Test 0)": "logs/test0_aggregated_metrics.json",
    "Orthogonal (Test 1)": "logs/test1_aggregated_metrics.json",
    "Orthogonal + Norm (Test 1B)": "logs/test1b_aggregated_metrics.json",
}


def load_metrics(path: str):
    with open(path, "r") as f:
        return json.load(f)


def aggregate_metrics(use_aggregated=False):
    """Aggregate metrics from either single-seed or multi-seed results."""
    log_dirs = AGGREGATED_LOG_DIRS if use_aggregated else LOG_DIRS
    names, bwt, fwt, ret, trans = [], [], [], [], []
    bwt_err, fwt_err, ret_err, trans_err = [], [], [], []

    for label, path in log_dirs.items():
        if not os.path.exists(path):
            print(f"Missing: {path}, skipping {label}.")
            continue

        metrics = load_metrics(path)

        if use_aggregated and "cl_metrics" in metrics:
            model1_metrics = metrics["cl_metrics"]["model1"]
            model2_metrics = metrics["cl_metrics"]["model2"]

            avg_bwt = (
                model1_metrics["BWT"]["mean"] + model2_metrics["BWT"]["mean"]
            ) / 2
            avg_fwt = (
                model1_metrics["FWT"]["mean"] + model2_metrics["FWT"]["mean"]
            ) / 2
            avg_ret = (
                model1_metrics["Retention%"]["mean"]
                + model2_metrics["Retention%"]["mean"]
            ) / 2
            avg_trans = (
                model1_metrics["Transfer%"]["mean"]
                + model2_metrics["Transfer%"]["mean"]
            ) / 2

            bwt_error = (
                (model1_metrics["BWT"]["ci_high"] - model1_metrics["BWT"]["ci_low"])
                + (model2_metrics["BWT"]["ci_high"] - model2_metrics["BWT"]["ci_low"])
            ) / 4
            fwt_error = (
                (model1_metrics["FWT"]["ci_high"] - model1_metrics["FWT"]["ci_low"])
                + (model2_metrics["FWT"]["ci_high"] - model2_metrics["FWT"]["ci_low"])
            ) / 4
            ret_error = (
                (
                    model1_metrics["Retention%"]["ci_high"]
                    - model1_metrics["Retention%"]["ci_low"]
                )
                + (
                    model2_metrics["Retention%"]["ci_high"]
                    - model2_metrics["Retention%"]["ci_low"]
                )
            ) / 4
            trans_error = (
                (
                    model1_metrics["Transfer%"]["ci_high"]
                    - model1_metrics["Transfer%"]["ci_low"]
                )
                + (
                    model2_metrics["Transfer%"]["ci_high"]
                    - model2_metrics["Transfer%"]["ci_low"]
                )
            ) / 4

            bwt_err.append(bwt_error)
            fwt_err.append(fwt_error)
            ret_err.append(ret_error)
            trans_err.append(trans_error)
        else:
            avg_bwt = (metrics["model1"]["BWT"] + metrics["model2"]["BWT"]) / 2
            avg_fwt = (metrics["model1"]["FWT"] + metrics["model2"]["FWT"]) / 2
            avg_ret = (
                metrics["model1"]["Retention%"] + metrics["model2"]["Retention%"]
            ) / 2
            avg_trans = (
                metrics["model1"]["Transfer%"] + metrics["model2"]["Transfer%"]
            ) / 2

            bwt_err.append(0)
            fwt_err.append(0)
            ret_err.append(0)
            trans_err.append(0)

        names.append(label)
        bwt.append(avg_bwt)
        fwt.append(avg_fwt)
        ret.append(avg_ret)
        trans.append(avg_trans)

    if use_aggregated:
        return names, bwt, fwt, ret, trans, bwt_err, fwt_err, ret_err, trans_err
    else:
        return names, bwt, fwt, ret, trans


def plot_comparison(
    names: List[str],
    bwt: List[float],
    fwt: List[float],
    ret: List[float],
    trans: List[float],
    bwt_err: List[float] = None,
    fwt_err: List[float] = None,
    ret_err: List[float] = None,
    trans_err: List[float] = None,
    wandb_run=None,
):
    """Plot comparison with optional error bars for multi-seed results."""
    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - 1.5 * width,
        bwt,
        width,
        label="BWT (Forgetting)",
        yerr=bwt_err if bwt_err else None,
        capsize=5,
    )
    ax.bar(
        x - 0.5 * width,
        fwt,
        width,
        label="FWT (Transfer)",
        yerr=fwt_err if fwt_err else None,
        capsize=5,
    )
    ax.bar(
        x + 0.5 * width,
        ret,
        width,
        label="Retention %",
        yerr=ret_err if ret_err else None,
        capsize=5,
    )
    ax.bar(
        x + 1.5 * width,
        trans,
        width,
        label="Transfer %",
        yerr=trans_err if trans_err else None,
        capsize=5,
    )

    ax.set_ylabel("Metric Value")
    title = "Continual Learning Metric Comparison"
    if bwt_err:
        title += " (with 95% Confidence Intervals)"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs("logs", exist_ok=True)
    filename = (
        "logs/cl_comparison_plot_with_ci.png"
        if bwt_err
        else "logs/cl_comparison_plot.png"
    )
    plt.savefig(filename, dpi=300)

    if wandb_run and wandb:
        plot_name = "comparison_plot_with_ci" if bwt_err else "comparison_plot"
        wandb_run.log({plot_name: wandb.Image(plt)})

    plt.show()
