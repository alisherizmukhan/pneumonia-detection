"""Interpretability analysis: do models focus on lungs or background?

Decision rules:
  avg lung-focus score >= 0.60  →  GOOD (focuses on lungs)
  avg lung-focus score <  0.40  →  BAD  (focuses on background / uses shortcuts)
  otherwise                     →  MIXED

The lung region is approximated as the inner 80% × 70% of the image,
which covers the bilateral lung fields in a standard PA chest X-ray.
"""

import os
import sys
import json
import csv
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import get_logger, load_config


LUNG_REGION = {
    "y_start": 0.15,
    "y_end":   0.85,
    "x_start": 0.10,
    "x_end":   0.90,
}


def lung_focus_score(heatmap: np.ndarray) -> float:
    """Fraction of total attribution mass that falls inside the lung region."""
    H, W = heatmap.shape
    y0 = int(LUNG_REGION["y_start"] * H)
    y1 = int(LUNG_REGION["y_end"] * H)
    x0 = int(LUNG_REGION["x_start"] * W)
    x1 = int(LUNG_REGION["x_end"] * W)

    total = heatmap.sum()
    if total == 0:
        return 0.0
    inside = heatmap[y0:y1, x0:x1].sum()
    return float(inside / total)


def analyze_model(model_name: str, results_dir: str, logger) -> dict:
    """Compute per-method lung-focus scores and assign a verdict."""
    interp_dir = os.path.join(results_dir, "interpretability", model_name)
    if not os.path.isdir(interp_dir):
        logger.warning(f"No interpretability results for {model_name}")
        return {
            "model": model_name,
            "gradcam_lung_focus": None,
            "lrp_lung_focus": None,
            "occlusion_lung_focus": None,
            "avg_lung_focus": None,
            "verdict": "NO_DATA",
        }

    gradcam_scores, lrp_scores, occlusion_scores = [], [], []

    for img_dir in sorted(os.listdir(interp_dir)):
        img_path = os.path.join(interp_dir, img_dir)
        if not os.path.isdir(img_path):
            continue
        for method_file, score_list in [
            ("gradcam.npy",   gradcam_scores),
            ("lrp.npy",       lrp_scores),
            ("occlusion.npy", occlusion_scores),
        ]:
            npy_path = os.path.join(img_path, method_file)
            if os.path.exists(npy_path):
                heatmap = np.load(npy_path)
                score_list.append(lung_focus_score(heatmap))

    def mean_or_none(lst):
        return float(np.mean(lst)) if lst else None

    gradcam_mean   = mean_or_none(gradcam_scores)
    lrp_mean       = mean_or_none(lrp_scores)
    occlusion_mean = mean_or_none(occlusion_scores)

    available = [s for s in [gradcam_mean, lrp_mean, occlusion_mean] if s is not None]
    if available:
        avg = float(np.mean(available))
        if avg >= 0.60:
            verdict = "GOOD — focuses on lungs"
        elif avg < 0.40:
            verdict = "BAD — focuses on background"
        else:
            verdict = "MIXED — partial lung focus"
    else:
        avg = None
        verdict = "NO_DATA"

    avg_str = f"{avg:.3f}" if avg is not None else "N/A"
    logger.info(f"  {model_name}: avg_lung_focus={avg_str}  →  {verdict}")

    return {
        "model": model_name,
        "gradcam_lung_focus": gradcam_mean,
        "lrp_lung_focus":     lrp_mean,
        "occlusion_lung_focus": occlusion_mean,
        "avg_lung_focus": avg,
        "verdict": verdict,
    }


def run_analysis(config: dict) -> list:
    logger = get_logger("analysis")
    models_list = config.get(
        "models", ["densenet121", "resnet18", "efficientnet_b0", "mobilenet_v2"]
    )
    results_dir = config.get("results_dir", "results")

    logger.info("=== Interpretability Analysis ===")
    logger.info("Rule: avg lung-focus >= 0.60 → GOOD  |  < 0.40 → BAD  |  else → MIXED")

    verdicts = [analyze_model(m, results_dir, logger) for m in models_list]

    analysis_path = os.path.join(results_dir, "analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(verdicts, f, indent=2)
    logger.info(f"Analysis saved: {analysis_path}")

    _update_comparison_csv(verdicts, results_dir, logger)

    logger.info("=== Analysis complete ===")
    return verdicts


def _update_comparison_csv(verdicts: list, results_dir: str, logger) -> None:
    """Add GradCAM, LRP, Occlusion, and Verdict columns to final_comparison.csv."""
    csv_path = os.path.join(results_dir, "final_comparison.csv")
    if not os.path.exists(csv_path):
        logger.warning("final_comparison.csv not found — run evaluate.py first")
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        existing_fields = list(reader.fieldnames or [])

    verdict_map = {v["model"]: v for v in verdicts}

    new_fields = existing_fields[:]
    for col in ["GradCAM", "LRP", "Occlusion", "Verdict"]:
        if col not in new_fields:
            new_fields.append(col)

    def fmt(val):
        return f"{val:.3f}" if val is not None else ""

    for row in rows:
        v = verdict_map.get(row.get("Model", ""), {})
        row["GradCAM"]  = fmt(v.get("gradcam_lung_focus"))
        row["LRP"]      = fmt(v.get("lrp_lung_focus"))
        row["Occlusion"] = fmt(v.get("occlusion_lung_focus"))
        row["Verdict"]  = v.get("verdict", "")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Updated final_comparison.csv with interpretability verdicts")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze interpretability maps and assign lung-focus verdicts"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_analysis(config)


if __name__ == "__main__":
    main()
