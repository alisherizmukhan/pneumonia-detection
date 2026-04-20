import os
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm

from data import get_dataloaders, compute_class_weights
from models import get_model
from utils import (
    set_seed, save_model, get_device, get_logger, load_config,
    save_metrics, EarlyStopping, create_run_dir, save_run_config,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate model. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            labels = labels.float().to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_training_history(history, save_path):
    """Plot and save training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def train(config):
    """Full training pipeline."""
    logger = get_logger("train")

    set_seed(config.get("seed", 42))
    device = get_device()
    logger.info(f"Using device: {device}")

    # Experiment tracking — create run directory
    results_dir = config.get("results_dir", "results")
    run_dir = create_run_dir(results_dir)
    save_run_config(config, run_dir)
    logger.info(f"Run directory: {run_dir}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config.get("num_workers", 4),
    )

    # Class weights for imbalanced data
    pos_weight, class_counts = compute_class_weights(config["data_dir"])
    pos_weight = pos_weight.to(device)
    logger.info(f"Class counts — NORMAL: {class_counts[0]}, PNEUMONIA: {class_counts[1]}")
    logger.info(f"Using pos_weight: {pos_weight.item():.4f}")

    # Model
    model_name = config.get("model", "densenet121")
    model = get_model(
        model_name=model_name,
        freeze_backbone=config.get("freeze_backbone", False),
    )
    model = model.to(device)
    logger.info(f"Model: {model_name}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Learning rate scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config.get("lr_step_size", 5),
        gamma=config.get("lr_gamma", 0.1),
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.get("early_stopping_patience", 3),
    )

    # Training loop
    best_val_loss = float("inf")
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()

        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch {epoch}/{config['epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, f"best_model_{model_name}.pt")
            save_model(model, best_path)
            logger.info(f"Saved best model (val_loss={val_loss:.4f})")

        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Save history
    history_path = os.path.join(results_dir, "history.json")
    save_metrics(history, history_path)

    # Also save to run directory
    save_metrics(history, os.path.join(run_dir, "history.json"))

    # Save training plot to both locations
    plot_path = os.path.join(results_dir, "training_plot.png")
    plot_training_history(history, plot_path)
    plot_training_history(history, os.path.join(run_dir, "training_plot.png"))
    logger.info(f"Training plot saved to {plot_path}")

    # Save model info to run directory
    model_info = {
        "model_name": model_name,
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history["train_loss"]),
        "class_counts": {"NORMAL": class_counts[0], "PNEUMONIA": class_counts[1]},
    }
    save_metrics(model_info, os.path.join(run_dir, "model_info.json"))

    logger.info("Training complete.")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
