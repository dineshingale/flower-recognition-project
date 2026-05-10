import wandb

def init_wandb(project_name, config_dict):
    """Initialize Weights & Biases logging."""
    wandb.init(project=project_name, config=config_dict)
    return wandb.config
