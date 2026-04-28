"""Aggregate all experiment results into a single CSV for presentation.

Reads:
  results/comparison.json         — per-model test-set metrics (all 3 models)
  results/kfold/kfold_*.json      — k-fold CV summaries per model
  results/best_threshold.json     — threshold sweep for primary model

Writes:
  results/all_results.csv         — flat table, one row per (model, eval_type)
  results/kfold_summary.csv       — k-fold mean ± std table
  results/threshold_sweep.csv     — threshold sweep table

Usage:
    python src/collect_results.py
    python src/collect_results.py --results-dir results
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import json
import csv
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(rows: list[dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


def collect_test_metrics(results_dir: str) -> list[dict]:
    """Read comparison.json and return one row per model."""
    path = os.path.join(results_dir, "comparison.json")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return []

    data = load_json(path)
    rows = []
    for model_name, m in data.items():
        rows.append({
            "model": model_name,
            "eval_type": "test_set",
            "threshold": 0.5,
            "accuracy":  round(m.get("accuracy", float("nan")), 4),
            "precision": round(m.get("precision", float("nan")), 4),
            "recall":    round(m.get("recall", float("nan")), 4),
            "f1":        round(m.get("f1_score", float("nan")), 4),
            "f1_at_0.3": round(m.get("f1_at_0.3", float("nan")), 4),
            "roc_auc":   round(m.get("roc_auc", float("nan")), 4),
            "acc_mean":  "",
            "acc_std":   "",
            "f1_mean":   "",
            "f1_std":    "",
            "auc_mean":  "",
            "auc_std":   "",
        })
    return rows


def collect_kfold_metrics(results_dir: str) -> tuple[list[dict], list[dict]]:
    """Read kfold_*.json files. Returns (all_results_rows, kfold_summary_rows)."""
    kfold_dir = os.path.join(results_dir, "kfold")
    if not os.path.isdir(kfold_dir):
        print(f"  [skip] {kfold_dir} not found")
        return [], []

    all_rows = []
    summary_rows = []

    for fname in sorted(os.listdir(kfold_dir)):
        if not fname.startswith("kfold_") or not fname.endswith(".json"):
            continue
        data = load_json(os.path.join(kfold_dir, fname))
        model_name = data.get("model", fname.replace("kfold_", "").replace(".json", ""))
        n_folds = data.get("n_folds", "?")
        threshold = data.get("threshold", 0.5)

        acc  = data.get("accuracy", {})
        f1   = data.get("f1", {})
        auc  = data.get("roc_auc", {})

        all_rows.append({
            "model": model_name,
            "eval_type": f"kfold_{n_folds}fold",
            "threshold": threshold,
            "accuracy":  "",
            "precision": "",
            "recall":    "",
            "f1":        "",
            "f1_at_0.3": "",
            "roc_auc":   "",
            "acc_mean":  round(acc.get("mean", float("nan")), 4),
            "acc_std":   round(acc.get("std",  float("nan")), 4),
            "f1_mean":   round(f1.get("mean",  float("nan")), 4),
            "f1_std":    round(f1.get("std",   float("nan")), 4),
            "auc_mean":  round(auc.get("mean", float("nan")), 4),
            "auc_std":   round(auc.get("std",  float("nan")), 4),
        })

        summary_rows.append({
            "model": model_name,
            "n_folds": n_folds,
            "threshold": threshold,
            "accuracy":  f"{acc.get('mean', 0):.4f} ± {acc.get('std', 0):.4f}",
            "f1":        f"{f1.get('mean', 0):.4f} ± {f1.get('std', 0):.4f}",
            "roc_auc":   f"{auc.get('mean', 0):.4f} ± {auc.get('std', 0):.4f}",
        })

        # Per-fold breakdown
        for i, fold in enumerate(data.get("per_fold", []), 1):
            all_rows.append({
                "model": model_name,
                "eval_type": f"kfold_fold{i}",
                "threshold": threshold,
                "accuracy":  round(fold.get("accuracy", float("nan")), 4),
                "precision": "",
                "recall":    "",
                "f1":        round(fold.get("f1", float("nan")), 4),
                "f1_at_0.3": "",
                "roc_auc":   round(fold.get("roc_auc", float("nan")), 4),
                "acc_mean":  "",
                "acc_std":   "",
                "f1_mean":   "",
                "f1_std":    "",
                "auc_mean":  "",
                "auc_std":   "",
            })

    return all_rows, summary_rows


def collect_threshold_sweep(results_dir: str) -> list[dict]:
    """Read best_threshold.json and return one row per threshold."""
    path = os.path.join(results_dir, "best_threshold.json")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return []

    data = load_json(path)
    rows = []
    for entry in data.get("all_thresholds", []):
        rows.append({
            "threshold": entry.get("threshold"),
            "accuracy":  round(entry.get("accuracy", float("nan")), 4),
            "precision": round(entry.get("precision", float("nan")), 4),
            "recall":    round(entry.get("recall", float("nan")), 4),
            "f1":        round(entry.get("f1_score", float("nan")), 4),
            "roc_auc":   round(entry.get("roc_auc", float("nan")), 4),
            "is_best":   entry.get("threshold") == data.get("best_threshold"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Aggregate all results into CSV")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    rd = args.results_dir

    print("Collecting test-set metrics...")
    test_rows = collect_test_metrics(rd)

    print("Collecting k-fold CV metrics...")
    kfold_rows, kfold_summary = collect_kfold_metrics(rd)

    print("Collecting threshold sweep...")
    threshold_rows = collect_threshold_sweep(rd)

    # Combined flat table
    all_rows = test_rows + kfold_rows
    if all_rows:
        write_csv(all_rows, os.path.join(rd, "all_results.csv"))

    if kfold_summary:
        write_csv(kfold_summary, os.path.join(rd, "kfold_summary.csv"))

    if threshold_rows:
        write_csv(threshold_rows, os.path.join(rd, "threshold_sweep.csv"))

    print("\nDone. Output files:")
    for fname in ["all_results.csv", "kfold_summary.csv", "threshold_sweep.csv"]:
        fpath = os.path.join(rd, fname)
        if os.path.exists(fpath):
            print(f"  {fpath}")


if __name__ == "__main__":
    main()
