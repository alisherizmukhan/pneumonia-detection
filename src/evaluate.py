import os
import sys
import csv
import argparse

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from data import get_dataloaders
from models import get_model
from utils import set_seed, load_model, get_device, get_logger, load_config, save_metrics


def predict(model, loader, device):
    """Run inference and return (y_true, y_prob) as numpy arrays."""
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
    return np.array(all_labels), np.array(all_probs)


def compute_metrics(y_true, y_prob, threshold: float = 0.3) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "threshold": float(threshold),
    }


def evaluate_all(config: dict) -> dict:
    logger = get_logger("evaluate")
    set_seed(config.get("seed", 42))
    device = get_device()
    logger.info(f"Device: {device}")

    models_list = config.get(
        "models", ["densenet121", "resnet18", "efficientnet_b0", "mobilenet_v2"]
    )
    threshold = config.get("threshold", 0.3)
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    results_dir = config.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    _, _, test_loader = get_dataloaders(
        data_dir=config["data_dir"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        num_workers=config.get("num_workers", 4),
    )

    all_metrics = {}

    for model_name in models_list:
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path} — skipping {model_name}")
            continue

        logger.info(f"Evaluating {model_name}...")
        model = get_model(model_name)
        model = load_model(model, ckpt_path, device)
        model = model.to(device)

        y_true, y_prob = predict(model, test_loader, device)
        metrics = compute_metrics(y_true, y_prob, threshold)
        metrics["model"] = model_name

        save_metrics(metrics, os.path.join(results_dir, f"{model_name}_metrics.json"))
        logger.info(
            f"  {model_name}: ROC-AUC={metrics['roc_auc']:.4f}  "
            f"PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1_score']:.4f}  "
            f"Recall={metrics['recall']:.4f}  Acc={metrics['accuracy']:.4f}"
        )
        all_metrics[model_name] = metrics

    if all_metrics:
        _write_comparison_csv(all_metrics, results_dir, logger)

    logger.info("Evaluation complete.")
    return all_metrics


def _write_comparison_csv(all_metrics: dict, results_dir: str, logger) -> None:
    path = os.path.join(results_dir, "final_comparison.csv")
    fieldnames = ["Model", "ROC-AUC", "PR-AUC", "Recall", "F1", "Accuracy", "Threshold"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, m in all_metrics.items():
            writer.writerow({
                "Model": model_name,
                "ROC-AUC": f"{m['roc_auc']:.4f}",
                "PR-AUC": f"{m['pr_auc']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1": f"{m['f1_score']:.4f}",
                "Accuracy": f"{m['accuracy']:.4f}",
                "Threshold": m["threshold"],
            })
    logger.info(f"Comparison CSV saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate all pneumonia detection models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_all(config)


if __name__ == "__main__":
    main()
