import os
import argparse
import json
import shutil

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from tqdm import tqdm
from torchvision import datasets

from data import get_dataloaders
from models import get_model
from utils import set_seed, load_model, get_device, get_logger, load_config, save_metrics


def predict(model, loader, device):
    """Run inference and return true labels, predicted labels, and probabilities."""
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images = images.to(device)
            outputs = model(images).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    return all_labels, all_preds, all_probs


def compute_metrics(y_true, y_pred, y_prob):
    """Compute classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Normal", "Pneumonia"]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def extract_error_images(data_dir, y_true, y_pred, y_prob, results_dir, logger):
    """Copy misclassified images to results/errors/."""
    test_dir = os.path.join(data_dir, "test")
    test_dataset = datasets.ImageFolder(root=test_dir)
    samples = test_dataset.samples  # list of (path, class_idx)

    fp_dir = os.path.join(results_dir, "errors", "false_positive")
    fn_dir = os.path.join(results_dir, "errors", "false_negative")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    fp_count = 0
    fn_count = 0
    for i, (img_path, true_label) in enumerate(samples):
        pred_label = y_pred[i]
        if true_label == 0 and pred_label == 1:
            # False positive: predicted pneumonia but actually normal
            dst = os.path.join(fp_dir, f"fp_{fp_count:04d}_{os.path.basename(img_path)}")
            shutil.copy2(img_path, dst)
            fp_count += 1
        elif true_label == 1 and pred_label == 0:
            # False negative: predicted normal but actually pneumonia
            dst = os.path.join(fn_dir, f"fn_{fn_count:04d}_{os.path.basename(img_path)}")
            shutil.copy2(img_path, dst)
            fn_count += 1

    logger.info(f"Error analysis: {fp_count} false positives, {fn_count} false negatives")
    logger.info(f"Error images saved to {os.path.join(results_dir, 'errors')}")


def threshold_tuning(y_true, y_prob, results_dir, logger):
    """Test multiple thresholds and save the best one."""
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        metrics = compute_metrics(y_true, preds, y_prob)
        metrics["threshold"] = t
        results.append(metrics)
        logger.info(
            f"  Threshold {t:.1f} | Acc: {metrics['accuracy']:.4f} | "
            f"Prec: {metrics['precision']:.4f} | Rec: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f}"
        )

    # Select best threshold by F1 score
    best = max(results, key=lambda x: x["f1_score"])
    best_threshold_data = {
        "best_threshold": best["threshold"],
        "metrics_at_best": best,
        "all_thresholds": results,
    }

    save_path = os.path.join(results_dir, "best_threshold.json")
    save_metrics(best_threshold_data, save_path)
    logger.info(f"Best threshold: {best['threshold']} (F1={best['f1_score']:.4f})")
    logger.info(f"Threshold results saved to {save_path}")

    return best["threshold"]


def evaluate(config, model_path: str):
    """Full evaluation pipeline."""
    logger = get_logger("evaluate")

    set_seed(config.get("seed", 42))
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data
    _, _, test_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config.get("num_workers", 4),
    )

    # Model
    logger.info("Loading model: DenseNet121")
    logger.info(f"Checkpoint: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = get_model(
        freeze_backbone=config.get("freeze_backbone", False),
    )
    model = load_model(model, model_path, device)
    model = model.to(device)
    logger.info(f"Loaded model from {model_path}")

    # Predict
    y_true, y_pred, y_prob = predict(model, test_loader, device)

    # Metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    metrics_path = os.path.join(results_dir, "metrics.json")
    save_metrics(metrics, metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")

    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Plots
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, cm_path)
    logger.info(f"Confusion matrix saved to {cm_path}")

    roc_path = os.path.join(results_dir, "roc_curve.png")
    plot_roc_curve(y_true, y_prob, roc_path)
    logger.info(f"ROC curve saved to {roc_path}")

    # Error analysis
    logger.info("Running error analysis...")
    extract_error_images(
        config["data_dir"], y_true, y_pred, y_prob, results_dir, logger
    )

    # Threshold tuning
    logger.info("Running threshold tuning...")
    threshold_tuning(y_true, y_prob, results_dir, logger)

    logger.info("Evaluation complete.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate pneumonia detection model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--model", type=str, default="checkpoints/best_model.pt",
        help="Path to saved model checkpoint",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    evaluate(config, args.model)


if __name__ == "__main__":
    main()
