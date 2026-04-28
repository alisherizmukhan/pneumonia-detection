"""Stratified k-fold cross-validation on the training set.

Addresses the methodological weakness of the official 16-image validation split
(one misclassification = 6.25% accuracy swing) by performing model selection and
honest performance estimation entirely within the training set. The held-out test
set is never touched.

References:
    Kohavi (1995), "A Study of Cross-Validation and Bootstrap for Accuracy
    Estimation and Model Selection", IJCAI.

Usage:
    python src/kfold.py --config configs/config.yaml --folds 5
    python src/kfold.py --config configs/config_resnet18.yaml --folds 5
    python src/kfold.py --config configs/config_baseline.yaml --folds 5
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import time
from logging import Logger
from typing import Final

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from data import get_transforms, compute_class_weights
from models import get_model
from utils import set_seed, get_device, get_logger, load_config, save_metrics, EarlyStopping

DEFAULT_SEED: Final[int] = 42
DEFAULT_MODEL: Final[str] = "densenet121"
DEFAULT_THRESHOLD: Final[float] = 0.5
DEFAULT_IMAGE_SIZE: Final[int] = 224
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_NUM_WORKERS: Final[int] = 4
DEFAULT_EPOCHS: Final[int] = 10
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4
DEFAULT_LR_STEP_SIZE: Final[int] = 5
DEFAULT_LR_GAMMA: Final[float] = 0.1
DEFAULT_EARLY_STOPPING_PATIENCE: Final[int] = 3
DEFAULT_RESULTS_DIR: Final[str] = "results"

FoldMetrics = dict[str, float]
Summary = dict[str, object]


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns average loss over the epoch."""
    model.train()
    running_loss: float = 0.0
    total: int = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total


def eval_fold(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = DEFAULT_THRESHOLD,
) -> FoldMetrics:
    """Evaluate model on a fold. Returns accuracy, F1, and ROC-AUC."""
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images).squeeze(1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    y_true: np.ndarray = np.array(all_labels)
    y_prob: np.ndarray = np.array(all_probs)
    y_pred: np.ndarray = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":  float(roc_auc_score(y_true, y_prob)),
    }


def run_kfold(config: dict, n_folds: int = 5) -> Summary:
    """Run stratified k-fold CV and return aggregated summary."""
    logger: Logger = get_logger("kfold")

    seed: int = config.get("seed", DEFAULT_SEED)
    set_seed(seed)
    device: torch.device = get_device()
    logger.info(f"Device: {device}")

    model_name: str = config.get("model", DEFAULT_MODEL)
    threshold: float = config.get("threshold", DEFAULT_THRESHOLD)
    image_size: int = config.get("image_size", DEFAULT_IMAGE_SIZE)
    batch_size: int = config.get("batch_size", DEFAULT_BATCH_SIZE)
    num_workers: int = config.get("num_workers", DEFAULT_NUM_WORKERS)
    epochs: int = config.get("epochs", DEFAULT_EPOCHS)
    data_dir: str = config["data_dir"]
    train_dir: str = os.path.join(data_dir, "train")

    logger.info(f"Model: {model_name} | Folds: {n_folds} | Threshold: {threshold}")

    train_transform, val_transform = get_transforms(image_size)

    # Two datasets over the same files — different transforms applied per-fold role
    dataset_aug = datasets.ImageFolder(root=train_dir, transform=train_transform)
    dataset_clean = datasets.ImageFolder(root=train_dir, transform=val_transform)

    targets: list[int] = [label for _, label in dataset_aug.samples]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    pos_weight, class_counts = compute_class_weights(data_dir)
    pos_weight = pos_weight.to(device)
    logger.info(f"Class counts — NORMAL: {class_counts[0]}, PNEUMONIA: {class_counts[1]}")
    logger.info(f"pos_weight: {pos_weight.item():.4f}")

    fold_metrics: list[FoldMetrics] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_idx}/{n_folds} — train: {len(train_idx)}, val: {len(val_idx)}")

        train_loader = DataLoader(
            Subset(dataset_aug, train_idx),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(dataset_clean, val_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        model = get_model(
            model_name=model_name,
            freeze_backbone=config.get("freeze_backbone", False),
        )
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", DEFAULT_WEIGHT_DECAY),
        )
        scheduler = StepLR(
            optimizer,
            step_size=config.get("lr_step_size", DEFAULT_LR_STEP_SIZE),
            gamma=config.get("lr_gamma", DEFAULT_LR_GAMMA),
        )
        early_stopping = EarlyStopping(
            patience=config.get("early_stopping_patience", DEFAULT_EARLY_STOPPING_PATIENCE)
        )

        best_val_loss: float = float("inf")
        best_metrics: FoldMetrics | None = None

        for epoch in range(1, epochs + 1):
            t0: float = time.time()
            train_loss: float = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

            model.eval()
            val_loss: float = 0.0
            total: int = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.float().to(device)
                    outputs = model(images).squeeze(1)
                    val_loss += criterion(outputs, labels).item() * images.size(0)
                    total += labels.size(0)
            val_loss /= total

            elapsed: float = time.time() - t0
            logger.info(
                f"  Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = eval_fold(model, val_loader, device, threshold)

            if early_stopping(val_loss):
                logger.info(f"  Early stopping at epoch {epoch}")
                break

        fold_metrics.append(best_metrics)
        logger.info(
            f"Fold {fold_idx} best — Acc: {best_metrics['accuracy']:.4f} | "
            f"F1: {best_metrics['f1']:.4f} | AUC: {best_metrics['roc_auc']:.4f}"
        )

    accs: list[float] = [m["accuracy"] for m in fold_metrics]
    f1s:  list[float] = [m["f1"]       for m in fold_metrics]
    aucs: list[float] = [m["roc_auc"]  for m in fold_metrics]

    summary: Summary = {
        "model":    model_name,
        "n_folds":  n_folds,
        "threshold": threshold,
        "per_fold": fold_metrics,
        "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
        "f1":       {"mean": float(np.mean(f1s)),  "std": float(np.std(f1s))},
        "roc_auc":  {"mean": float(np.mean(aucs)), "std": float(np.std(aucs))},
    }

    logger.info("\n" + "=" * 50)
    logger.info(f"K-Fold CV Summary ({model_name}, k={n_folds})")
    logger.info(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info(f"  F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"  ROC-AUC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    out_dir: str = os.path.join(config.get("results_dir", DEFAULT_RESULTS_DIR), "kfold")
    os.makedirs(out_dir, exist_ok=True)
    out_path: str = os.path.join(out_dir, f"kfold_{model_name}.json")
    save_metrics(summary, out_path)
    logger.info(f"Results saved to {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Stratified k-fold CV on training set")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    config: dict = load_config(args.config)
    run_kfold(config, n_folds=args.folds)


if __name__ == "__main__":
    main()
