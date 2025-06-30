from src.utils.compare import aggregate_metrics, plot_comparison

try:
    import wandb
except ImportError:
    wandb = None

if __name__ == "__main__":
    wandb_run = None
    if wandb and wandb.run is not None:
        wandb_run = wandb.run

    names, bwt, fwt, ret, trans = aggregate_metrics()
    plot_comparison(names, bwt, fwt, ret, trans, wandb_run=wandb_run)
    print("✓ Comparison completed and plot saved.")
