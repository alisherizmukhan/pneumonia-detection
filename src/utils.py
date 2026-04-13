import os
import random
import logging
import json

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, path: str):
    """Save model state dict to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model, path: str, device: torch.device = None):
    """Load model state dict from path."""
    if device is None:
        device = torch.device("cpu")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Create and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def save_metrics(metrics: dict, path: str):
    """Save metrics dictionary to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_run_dir(results_dir: str) -> str:
    """Create a new numbered run directory under results_dir.

    Returns:
        Path to the new run directory (e.g., results/run_1).
    """
    os.makedirs(results_dir, exist_ok=True)
    existing = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("run_")
    ]
    run_nums = []
    for d in existing:
        try:
            run_nums.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            continue
    next_num = max(run_nums, default=0) + 1
    run_dir = os.path.join(results_dir, f"run_{next_num}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(config: dict, run_dir: str):
    """Save a copy of the config to the run directory."""
    import yaml

    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop
