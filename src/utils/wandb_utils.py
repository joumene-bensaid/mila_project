import os, warnings

try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(cfg, run_name: str):
    if wandb is None:
        return None
    try:
        key = os.getenv("WANDB_API_KEY")
        if key:
            wandb.login(key=key)
        else:
            wandb.login()
        return wandb.init(
            project="direction-aware-fusion", name=run_name, config=cfg.__dict__
        )
    except Exception as e:
        warnings.warn(f"WandB offline: {e}")
        return wandb.init(
            project="direction-aware-fusion",
            name=run_name + "_offline",
            mode="offline",
            config=cfg.__dict__,
        )
