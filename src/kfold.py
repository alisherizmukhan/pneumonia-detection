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
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from tqdm import tqdm

from data import get_transforms, compute_class_weights
from models import get_model
from utils import set_seed, get_device, get_logger, load_config, save_metrics, EarlyStopping


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0
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


def eval_fold(model, loader, device, threshold=0.5):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images).squeeze(1)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def run_kfold(config: dict, n_folds: int = 5):
    logger = get_logger("kfold")
    set_seed(config.get("seed", 42))
    device = get_device()
    logger.info(f"Device: {device}")

    model_name = config.get("model", "densenet121")
    threshold = config.get("threshold", 0.5)
    image_size = config.get("image_size", 224)
    batch_size = config.get("batch_size", 32)
    num_workers = config.get("num_workers", 4)
    epochs = config.get("epochs", 10)
    data_dir = config["data_dir"]
    train_dir = os.path.join(data_dir, "train")

    logger.info(f"Model: {model_name} | Folds: {n_folds} | Threshold: {threshold}")

    train_transform, val_transform = get_transforms(image_size)

    # Two datasets over the same files — different transforms applied per-fold role
    dataset_aug = datasets.ImageFolder(root=train_dir, transform=train_transform)
    dataset_clean = datasets.ImageFolder(root=train_dir, transform=val_transform)

    targets = [label for _, label in dataset_aug.samples]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.get("seed", 42))

    # Class weights from full training set (stable across folds)
    pos_weight, class_counts = compute_class_weights(data_dir)
    pos_weight = pos_weight.to(device)
    logger.info(f"Class counts — NORMAL: {class_counts[0]}, PNEUMONIA: {class_counts[1]}")
    logger.info(f"pos_weight: {pos_weight.item():.4f}")

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {fold_idx}/{n_folds} — train: {len(train_idx)}, val: {len(val_idx)}")

        train_loader = DataLoader(
            Subset(dataset_aug, train_idx),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(dataset_clean, val_idx),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )

        model = get_model(model_name=model_name, freeze_backbone=config.get("freeze_backbone", False))
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

        best_val_loss = float("inf")
        best_metrics = None

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            scheduler.step()

            # Compute val loss for early stopping
            model.eval()
            val_loss = 0.0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.float().to(device)
                    outputs = model(images).squeeze(1)
                    val_loss += criterion(outputs, labels).item() * images.size(0)
                    total += labels.size(0)
            val_loss /= total

            elapsed = time.time() - t0
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

    # Aggregate
    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]
    aucs = [m["roc_auc"] for m in fold_metrics]

    summary = {
        "model": model_name,
        "n_folds": n_folds,
        "threshold": threshold,
        "per_fold": fold_metrics,
        "accuracy":  {"mean": float(np.mean(accs)),  "std": float(np.std(accs))},
        "f1":        {"mean": float(np.mean(f1s)),   "std": float(np.std(f1s))},
        "roc_auc":   {"mean": float(np.mean(aucs)),  "std": float(np.std(aucs))},
    }

    logger.info("\n" + "=" * 50)
    logger.info(f"K-Fold CV Summary ({model_name}, k={n_folds})")
    logger.info(f"  Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    logger.info(f"  F1       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    logger.info(f"  ROC-AUC  : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    out_dir = os.path.join(config.get("results_dir", "results"), "kfold")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"kfold_{model_name}.json")
    save_metrics(summary, out_path)
    logger.info(f"Results saved to {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Stratified k-fold CV on training set")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    args = parser.parse_args()

    config = load_config(args.config)
    run_kfold(config, n_folds=args.folds)


if __name__ == "__main__":
    main()
