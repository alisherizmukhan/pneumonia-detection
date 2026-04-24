"""XAI comparison: GradCAM, Integrated Gradients (Captum), LRP (Zennit).

Uses production-grade libraries for IG and LRP rather than hand-rolled
implementations, avoiding all the hook-sharing / torch.compile / OOM issues
that arise when implementing these from scratch.

Libraries
---------
GradCAM          gradcam.py (existing, proven)
Integrated Grads captum.attr.IntegratedGradients  — Meta's reference impl
LRP              zennit  — original LRP authors' PyTorch library

Faithfulness metrics (doctor-free verification)
-----------------------------------------------
Deletion AUC   mask pixels by descending saliency; measure confidence drop.
               Lower = attribution correctly identifies critical regions.
Insertion AUC  reveal pixels into blurred baseline; measure confidence rise.
               Higher = better.

Usage
-----
    pip install captum zennit

    python src/xai_comparison.py \
        --config configs/config.yaml \
        --densenet-checkpoint checkpoints/best_model_densenet121.pt \
        --resnet-checkpoint  checkpoints/best_model_resnet18.pt \
        --data-dir data/chest_xray/test \
        --num-images 624 \
        --output results/xai_comparison

    # Skip faithfulness (visual only, fast)
    python src/xai_comparison.py --no-faithfulness

    # Force specific GPUs
    python src/xai_comparison.py --gpu 0 1
"""

import sys
import os
import json
import threading
import time
import warnings

sys.path.insert(0, os.path.dirname(__file__))

import argparse
from typing import Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms.functional import gaussian_blur

from data import get_transforms
from models import get_model
from utils import load_model, load_config
from gradcam import GradCAM, _get_target_layer, _denormalise, overlay_heatmap

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_SEED: Final[int]       = 42
DEFAULT_IMAGE_SIZE: Final[int] = 224
DEFAULT_NUM_IMAGES: Final[int] = 8
DEFAULT_IG_STEPS: Final[int]   = 300
DEFAULT_OUTPUT_DIR: Final[str] = "results/xai_comparison"
GRID_FIG_WIDTH: Final[int]     = 16
GRID_FIG_HEIGHT_PER_ROW: Final[int] = 4
GRID_DPI: Final[int]           = 150
BLUR_KERNEL: Final[int]        = 51
N_FAITH_STEPS: Final[int]      = 100
FAITH_CHUNK: Final[int]        = 25   # images per faithfulness forward chunk


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_gpu_devices() -> list:
    n = torch.cuda.device_count()
    if n == 0:
        print("[WARN] No CUDA devices — running on CPU.")
        return [torch.device("cpu")]
    devs = [torch.device(f"cuda:{i}") for i in range(n)]
    for d in devs:
        p = torch.cuda.get_device_properties(d)
        print(f"  {d}: {p.name}  {p.total_memory/1e9:.1f} GB VRAM")
    return devs


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients via Captum
# ─────────────────────────────────────────────────────────────────────────────

def compute_ig(
    model: nn.Module,
    inp: torch.Tensor,
    n_steps: int = DEFAULT_IG_STEPS,
    internal_batch_size: int = 32,
) -> np.ndarray:
    """Compute IG attribution using Captum's reference implementation.

    Captum handles chunking, baseline, gradient accumulation, and the
    Completeness axiom check internally.  internal_batch_size=32 caps
    peak activation memory regardless of n_steps or model depth.

    Returns (H, W) float32 array in [0, 1].
    """
    from captum.attr import IntegratedGradients

    model.eval()
    ig = IntegratedGradients(model)

    # Captum expects requires_grad on input
    inp_req = inp.float().requires_grad_(True)

    # target=None → attribute w.r.t. the raw scalar output (binary classifier)
    attribution = ig.attribute(
        inp_req,
        baselines=torch.zeros_like(inp_req),
        n_steps=n_steps,
        method="gausslegendre",          # more accurate than riemann
        internal_batch_size=internal_batch_size,
    )  # (1, C, H, W)

    attr = attribution.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()  # (H, W)
    a_min, a_max = attr.min(), attr.max()
    if a_max > a_min:
        attr = (attr - a_min) / (a_max - a_min)
    else:
        attr = np.zeros_like(attr)
    return attr


# ─────────────────────────────────────────────────────────────────────────────
# LRP via Zennit
# ─────────────────────────────────────────────────────────────────────────────

def compute_lrp(
    model: nn.Module,
    inp: torch.Tensor,
) -> np.ndarray:
    """Compute LRP attribution using Zennit.

    Uses EpsilonPlusFlat composite — appropriate for CNNs without residual
    connections (ResNet18) and with dense connections (DenseNet121).
    Zennit handles hook registration and cleanup internally via context manager,
    so there is zero risk of hooks persisting across calls.

    Returns (H, W) float32 array in [0, 1].
    """
    from zennit.composites import EpsilonPlusFlat
    from zennit.attribution import Gradient

    model.eval()
    inp_f = inp.float()

    composite = EpsilonPlusFlat()
    with Gradient(model=model, composite=composite) as attributor:
        # eye(1) because we have a single binary output
        out, relevance = attributor(inp_f, torch.ones(1, 1, device=inp.device))

    attr = relevance.abs().sum(dim=1).squeeze(0).detach().cpu().numpy()
    a_min, a_max = attr.min(), attr.max()
    if a_max > a_min:
        attr = (attr - a_min) / (a_max - a_min)
    else:
        attr = np.zeros_like(attr)
    return attr


# ─────────────────────────────────────────────────────────────────────────────
# Faithfulness metrics
# ─────────────────────────────────────────────────────────────────────────────

def _make_masked_batch(
    input_tensor: torch.Tensor,
    flat_indices: np.ndarray,
    n_steps: int,
    mode: str,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """Vectorised: build (n_steps+1, C, H, W) batch — no Python loop."""
    C, H, W  = input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]
    n_pixels  = H * W
    step_size = max(1, n_pixels // n_steps)
    N         = n_steps + 1

    # pixel_rank[p] = saliency rank of pixel p (0 = most salient)
    pixel_rank = np.empty(n_pixels, dtype=np.int64)
    pixel_rank[flat_indices] = np.arange(n_pixels)

    thresholds = np.arange(N, dtype=np.int64) * step_size   # (N,)
    mask = pixel_rank[None, :] < thresholds[:, None]         # (N, H*W) bool

    mask_t = torch.from_numpy(mask).to(input_tensor.device)  # (N, H*W)
    src    = input_tensor.view(1, C, n_pixels).expand(N, -1, -1)
    base   = baseline.view(1, C, n_pixels).expand(N, -1, -1)

    if mode == "insertion":
        result = torch.where(mask_t.unsqueeze(1), src, base)
    else:
        result = torch.where(mask_t.unsqueeze(1), base, src)

    return result.view(N, C, H, W)


def _batched_forward(
    model: nn.Module,
    batch: torch.Tensor,
    chunk: int = FAITH_CHUNK,
) -> np.ndarray:
    """Forward pass in chunks to bound VRAM; returns sigmoid probabilities."""
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, batch.shape[0], chunk):
            c = batch[s: s + chunk]
            with torch.autocast(
                device_type=c.device.type, dtype=torch.float16, enabled=c.is_cuda
            ):
                logits = model(c.float()).squeeze(-1)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def deletion_auc(
    model: nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    n_steps: int = N_FAITH_STEPS,
) -> float:
    flat_indices = np.argsort(saliency.ravel())[::-1]
    baseline = torch.zeros_like(input_tensor)
    batch    = _make_masked_batch(input_tensor, flat_indices, n_steps, "deletion", baseline)
    scores   = _batched_forward(model, batch.to(input_tensor.device))
    return float(np.trapezoid(scores, np.linspace(0, 1, len(scores))))


def insertion_auc(
    model: nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    n_steps: int = N_FAITH_STEPS,
) -> float:
    flat_indices = np.argsort(saliency.ravel())[::-1]
    blurred  = gaussian_blur(input_tensor.squeeze(0), [BLUR_KERNEL, BLUR_KERNEL]).unsqueeze(0)
    batch    = _make_masked_batch(input_tensor, flat_indices, n_steps, "insertion", blurred)
    scores   = _batched_forward(model, batch.to(input_tensor.device))
    return float(np.trapezoid(scores, np.linspace(0, 1, len(scores))))


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _save_xai_grid(rows: list, output_path: str, model_name: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(GRID_FIG_WIDTH, GRID_FIG_HEIGHT_PER_ROW * n))
    if n == 1:
        axes = [axes]
    col_titles = ["Original", "Grad-CAM", "Integrated Gradients", "LRP (ε+flat)"]

    for i, row in enumerate(rows):
        raw, maps  = row["raw"], [None, row["gradcam"], row["ig"], row["lrp"]]
        del_s, ins_s = row["deletion_auc"], row["insertion_auc"]

        for j, (ax, smap) in enumerate(zip(axes[i], maps)):
            if j == 0:
                ax.imshow(raw)
                ax.set_ylabel(row["label"], rotation=0, labelpad=55, fontsize=8, va="center")
            else:
                ax.imshow(raw, alpha=0.35)
                im = ax.imshow(smap, cmap="jet", vmin=0, vmax=1, alpha=0.65)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10, fontweight="bold")
            ax.axis("off")

        for j, key in enumerate(["gradcam", "ig", "lrp"]):
            axes[i][j+1].set_xlabel(
                f"Del↓={del_s.get(key, float('nan')):.3f}  "
                f"Ins↑={ins_s.get(key, float('nan')):.3f}",
                fontsize=7, color="navy",
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.colorbar(im, ax=[axes[-1][j] for j in range(1, 4)],
                     fraction=0.015, pad=0.04, label="Attribution intensity")
        fig.suptitle(
            f"XAI Comparison — {model_name}\n"
            "Del↓ = deletion AUC (lower = faithful)   "
            "Ins↑ = insertion AUC (higher = faithful)",
            fontsize=10, y=1.01,
        )
        plt.tight_layout()
    plt.savefig(output_path, dpi=GRID_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def _save_faithfulness_summary(summary: dict, output_dir: str) -> None:
    models, methods = list(summary.keys()), ["gradcam", "ig", "lrp"]
    m_labels = ["Grad-CAM", "Integ. Grad.", "LRP"]
    x, width = np.arange(len(methods)), 0.35
    fig, (ax_del, ax_ins) = plt.subplots(1, 2, figsize=(12, 5))
    colours = ["#1565C0", "#B71C1C", "#2E7D32"]

    for mi, (mn, colour) in enumerate(zip(models, colours)):
        offset  = (mi - (len(models)-1)/2) * width
        d_means = [np.mean(summary[mn][m]["deletion"])  for m in methods]
        i_means = [np.mean(summary[mn][m]["insertion"]) for m in methods]
        ax_del.bar(x + offset, d_means, width, label=mn, color=colour, alpha=0.8)
        ax_ins.bar(x + offset, i_means, width, label=mn, color=colour, alpha=0.8)

    for ax, title in [(ax_del, "Deletion AUC (↓ lower = more faithful)"),
                      (ax_ins, "Insertion AUC (↑ higher = more faithful)")]:
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(m_labels)
        ax.set_ylim(0, 1); ax.legend(); ax.set_ylabel("AUC"); ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Faithfulness Metrics — Mean over test images", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "faithfulness_summary.png")
    plt.savefig(path, dpi=GRID_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _save_consistency_scatter(summary: dict, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colours = {"densenet121": "#1565C0", "resnet18": "#B71C1C"}
    for ax, (m1, m2, label) in zip(axes, [("gradcam","ig","GC vs IG"),
                                           ("gradcam","lrp","GC vs LRP")]):
        for mn, c in colours.items():
            if mn not in summary: continue
            ax.scatter(summary[mn][m1]["deletion"], summary[mn][m2]["deletion"],
                       label=mn, color=c, alpha=0.7, s=60)
        ax.plot([0,1],[0,1],"k--",alpha=0.4,label="perfect agreement")
        ax.set_xlabel(f"{m1.upper()} Del AUC"); ax.set_ylabel(f"{m2.upper()} Del AUC")
        ax.set_title(f"{label}\n(near diagonal → methods agree)")
        ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)
    fig.suptitle("Cross-method consistency (Deletion AUC)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, "consistency_scatter.png")
    plt.savefig(path, dpi=GRID_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _save_metrics_json(summary: dict, output_dir: str) -> None:
    out = {}
    for mn, faith in summary.items():
        out[mn] = {}
        for method in ["gradcam", "ig", "lrp"]:
            d, ins = faith[method]["deletion"], faith[method]["insertion"]
            out[mn][method] = {
                "deletion_mean":       float(np.mean(d))   if d else None,
                "deletion_std":        float(np.std(d))    if d else None,
                "insertion_mean":      float(np.mean(ins)) if ins else None,
                "insertion_std":       float(np.std(ins))  if ins else None,
                "deletion_per_image":  d,
                "insertion_per_image": ins,
            }
    path = os.path.join(output_dir, "faithfulness_scores.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-model worker
# ─────────────────────────────────────────────────────────────────────────────

def run_xai_for_model(
    model: nn.Module,
    model_name: str,
    data_dir: str,
    image_size: int,
    num_images: int,
    output_dir: str,
    device: torch.device,
    seed: int = DEFAULT_SEED,
    compute_faithfulness: bool = True,
    ig_steps: int = DEFAULT_IG_STEPS,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    _, test_transform = get_transforms(image_size)
    dataset     = datasets.ImageFolder(root=data_dir, transform=test_transform)
    class_names = dataset.classes

    by_class: dict = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    per_class = max(1, num_images // len(class_names))
    rng       = np.random.default_rng(seed)
    chosen    = []
    for ci, cn in enumerate(class_names):
        sel = rng.choice(by_class[ci], size=min(per_class, len(by_class[ci])), replace=False)
        chosen.extend([(int(i), cn) for i in sel])

    gc_exp = GradCAM(model, _get_target_layer(model, model_name))

    faith: dict = {m: {"deletion": [], "insertion": []} for m in ["gradcam", "ig", "lrp"]}
    rows:  list = []

    print(f"\n[{model_name} @ {device}] {len(chosen)} images | IG steps={ig_steps} | faith={compute_faithfulness}")

    for si, (idx, class_name) in enumerate(chosen):
        tensor, _ = dataset[idx]
        inp = tensor.unsqueeze(0).to(device, non_blocking=True)
        raw = _denormalise(tensor)

        print(f"  [{si+1}/{len(chosen)}] {class_name}", end=" ", flush=True)

        # GradCAM — existing proven implementation
        gc_map = gc_exp(inp)

        # Integrated Gradients — Captum
        ig_map = compute_ig(model, inp, n_steps=ig_steps)

        # LRP — Zennit (context manager cleans up hooks automatically)
        lrp_map = compute_lrp(model, inp)

        # Free any cached memory before faithfulness batches
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print("attrs✓", end=" ", flush=True)

        del_scores: dict = {}
        ins_scores: dict = {}

        if compute_faithfulness:
            for key, smap in [("gradcam", gc_map), ("ig", ig_map), ("lrp", lrp_map)]:
                d   = deletion_auc(model, inp.detach(), smap)
                ins = insertion_auc(model, inp.detach(), smap)
                del_scores[key] = d
                ins_scores[key] = ins
                faith[key]["deletion"].append(d)
                faith[key]["insertion"].append(ins)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print("faith✓")
        else:
            del_scores = {k: float("nan") for k in ["gradcam", "ig", "lrp"]}
            ins_scores = {k: float("nan") for k in ["gradcam", "ig", "lrp"]}
            print()

        rows.append({
            "raw": raw, "gradcam": gc_map, "ig": ig_map, "lrp": lrp_map,
            "label": class_name, "deletion_auc": del_scores, "insertion_auc": ins_scores,
        })

    grid_path = os.path.join(output_dir, f"xai_{model_name}_combined.png")
    _save_xai_grid(rows, grid_path, model_name)
    gc_exp.remove_hooks()
    return faith


# ─────────────────────────────────────────────────────────────────────────────
# Parallel dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def run_all_models_parallel(
    checkpoints: dict,
    devices: list,
    data_dir: str,
    image_size: int,
    num_images: int,
    output_dir: str,
    seed: int,
    compute_faithfulness: bool,
    ig_steps: int,
) -> dict:
    results: dict = {}
    errors:  dict = {}
    lock = threading.Lock()

    def worker(model_name: str, ckpt_path: str, device: torch.device) -> None:
        try:
            model = get_model(model_name=model_name, freeze_backbone=False, pretrained=False)
            model = load_model(model, ckpt_path, device).to(device)
            faith = run_xai_for_model(
                model=model, model_name=model_name, data_dir=data_dir,
                image_size=image_size, num_images=num_images,
                output_dir=output_dir, device=device, seed=seed,
                compute_faithfulness=compute_faithfulness, ig_steps=ig_steps,
            )
            with lock:
                results[model_name] = faith
        except Exception as exc:
            import traceback
            with lock:
                errors[model_name] = str(exc)
            print(f"\n[ERROR] {model_name}: {exc}")
            traceback.print_exc()

    valid   = {mn: ck for mn, ck in checkpoints.items() if os.path.exists(ck)}
    for mn in set(checkpoints) - set(valid):
        print(f"[SKIP] Checkpoint not found: {checkpoints[mn]}")
    if not valid:
        print("[ERROR] No valid checkpoints."); return {}

    assignments = {mn: devices[i % len(devices)] for i, mn in enumerate(valid)}
    parallel    = len(valid) > 1 and len(devices) > 1

    if parallel:
        print(f"\nRunning {len(valid)} models in parallel across {len(devices)} GPUs:")
        for mn, dev in assignments.items():
            print(f"  {mn} → {dev}")
        threads = [
            threading.Thread(target=worker, args=(mn, valid[mn], assignments[mn]),
                             name=f"xai-{mn}", daemon=True)
            for mn in valid
        ]
        for t in threads: t.start()
        for t in threads: t.join()
    else:
        for mn, ck in valid.items():
            worker(mn, ck, assignments[mn])

    if errors:
        print(f"\n[WARN] {len(errors)} model(s) failed: {list(errors)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="XAI comparison: GradCAM | Captum IG | Zennit LRP"
    )
    parser.add_argument("--config",              default="configs/config.yaml")
    parser.add_argument("--densenet-checkpoint", default="checkpoints/best_model_densenet121.pt")
    parser.add_argument("--resnet-checkpoint",   default="checkpoints/best_model_resnet18.pt")
    parser.add_argument("--data-dir",            default=None)
    parser.add_argument("--num-images",          type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--output",              default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ig-steps",            type=int, default=DEFAULT_IG_STEPS)
    parser.add_argument("--no-faithfulness",     action="store_true")
    parser.add_argument("--gpu",                 type=int, nargs="+", default=None)
    args = parser.parse_args()

    enable_tf32()
    config     = load_config(args.config)
    image_size = config.get("image_size", DEFAULT_IMAGE_SIZE)
    data_dir   = args.data_dir or os.path.join(config["data_dir"], "test")
    seed       = config.get("seed", DEFAULT_SEED)
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "="*60)
    print(" XAI Comparison — GradCAM | Captum IG | Zennit LRP")
    print("="*60)
    all_devs = get_gpu_devices()
    devices  = [all_devs[i] for i in args.gpu if i < len(all_devs)] if args.gpu else all_devs

    print(f"\nData   : {data_dir}")
    print(f"Images : {args.num_images}  |  IG steps : {args.ig_steps}")
    print(f"GPUs   : {[str(d) for d in devices]}")
    print(f"Faithfulness: {not args.no_faithfulness}")

    checkpoints = {
        "densenet121": args.densenet_checkpoint,
        "resnet18":    args.resnet_checkpoint,
    }

    t_start = time.time()
    summary = run_all_models_parallel(
        checkpoints=checkpoints, devices=devices, data_dir=data_dir,
        image_size=image_size, num_images=args.num_images, output_dir=args.output,
        seed=seed, compute_faithfulness=not args.no_faithfulness, ig_steps=args.ig_steps,
    )
    elapsed = time.time() - t_start

    if not args.no_faithfulness and summary:
        print("\nGenerating summary plots...")
        _save_faithfulness_summary(summary, args.output)
        if len(summary) >= 2:
            _save_consistency_scatter(summary, args.output)
        _save_metrics_json(summary, args.output)

    print("\n" + "="*70)
    print(f"{'Model':<15} {'Method':<22} {'Del AUC (↓)':<15} {'Ins AUC (↑)':<15}")
    print("-"*70)
    for mn, faith in summary.items():
        for method in ["gradcam", "ig", "lrp"]:
            d, ins = faith[method]["deletion"], faith[method]["insertion"]
            if d:
                print(f"{mn:<15} {method:<22} {np.mean(d):<15.4f} {np.mean(ins):<15.4f}")
    print("="*70)
    print(f"\nTotal wall time: {elapsed:.1f}s")
    print("\nInterpretation guide:")
    print("  Deletion AUC  ↓  lower  = attribution covers regions model truly relies on")
    print("  Insertion AUC ↑  higher = revealing salient pixels quickly restores confidence")
    print("  Consistency       GC, IG, LRP deletion AUCs similar → XAI is coherent")
    print(f"\nOutputs: {args.output}/")


if __name__ == "__main__":
    main()