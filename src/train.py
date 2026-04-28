import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data import get_dataloaders, compute_class_weights
from models import get_model
from utils import (
    set_seed, save_model, get_device, get_logger,
    load_config, save_metrics, EarlyStopping,
)


def train_one_epoch(model, loader, criterion, optimizer, device):
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
    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
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
    return running_loss / total, correct / total


def plot_training_history(history, save_path):
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
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def train(model_name: str, config: dict):
    logger = get_logger(f"train.{model_name}")
    set_seed(config.get("seed", 42))
    device = get_device()
    logger.info(f"Training {model_name} on {device}")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config.get("num_workers", 4),
    )

    pos_weight, class_counts = compute_class_weights(config["data_dir"])
    pos_weight = pos_weight.to(device)
    logger.info(
        f"NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}, "
        f"pos_weight={pos_weight.item():.4f}"
    )

    model = get_model(model_name)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )
    scheduler = StepLR(
        optimizer,
        step_size=config.get("lr_step_size", 5),
        gamma=config.get("lr_gamma", 0.1),
    )
    early_stopping = EarlyStopping(patience=config.get("early_stopping_patience", 3))

    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    results_dir = config.get("results_dir", "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_val_loss = float("inf")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "lr": [],
    }

    for epoch in range(1, config["epochs"] + 1):
        start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
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
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
            save_model(model, ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    save_metrics(history, os.path.join(results_dir, f"{model_name}_history.json"))
    plot_training_history(history, os.path.join(results_dir, f"{model_name}_training_plot.png"))
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train a pneumonia detection model")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name: densenet121, resnet18, efficientnet_b0, mobilenet_v2",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    train(args.model, config)


if __name__ == "__main__":
    main()
