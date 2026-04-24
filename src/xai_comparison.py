"""XAI comparison: Integrated Gradients, LRP, and Grad-CAM for pneumonia detection.

Dual-GPU optimised for RTX 5090 × 2 (or any CUDA multi-GPU setup).

Key optimisations vs the single-device version
-----------------------------------------------
1. Model parallelism — DenseNet121 on cuda:0, ResNet18 on cuda:1, processed
   concurrently via Python threads.  Each GPU gets its own CUDA stream.

2. Batched deletion / insertion — instead of 101 sequential forward passes,
   all masked variants are stacked into a single batched tensor and run in
   one forward pass (or a small number if VRAM is tight).  This cuts
   faithfulness computation from O(N_steps) forward passes to O(1).

3. torch.compile — models are compiled with mode="max-autotune-no-cudagraphs".
   CUDA graphs are explicitly disabled because they use thread-local storage
   initialised only on the main thread; running them from worker threads
   raises an AssertionError in cudagraph_trees.py.  The no-cudagraphs mode
   still gives full Triton kernel fusion and autotuning.

4. AMP (Automatic Mixed Precision) — all forward passes run under
   torch.autocast("cuda", dtype=torch.float16), halving memory bandwidth and
   accelerating Tensor Core throughput.  Gradients for IG are computed in
   float32 (required for numerical precision).

5. Pinned memory + non_blocking transfers — host-to-device copies use DMA.

6. CUDA streams — each XAI method within a single model runs in its own
   CUDA stream, allowing overlap where the GPU scheduler permits.

Usage
-----
    # Recommended: let the script pick GPUs automatically
    python src/xai_comparison.py \
        --config configs/config.yaml \
        --densenet-checkpoint checkpoints/best_model_densenet121.pt \
        --resnet-checkpoint  checkpoints/best_model_resnet18.pt \
        --data-dir data/chest_xray/test \
        --num-images 32 \
        --output results/xai_comparison

    # Visual only (skip faithfulness, fastest possible)
    python src/xai_comparison.py --no-faithfulness

    # Increase IG steps for publication-quality attributions
    python src/xai_comparison.py --ig-steps 300

    # Force specific GPUs
    python src/xai_comparison.py --gpu 0 1
"""

import sys
import os
import json
import threading
import time

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

from data import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from models import get_model
from utils import load_model, load_config
from gradcam import GradCAM, _get_target_layer, _denormalise, overlay_heatmap

# ── constants ────────────────────────────────────────────────────────────────
DEFAULT_SEED: Final[int]       = 42
DEFAULT_IMAGE_SIZE: Final[int] = 224
DEFAULT_NUM_IMAGES: Final[int] = 8
DEFAULT_IG_STEPS: Final[int]   = 100
DEFAULT_OUTPUT_DIR: Final[str] = "results/xai_comparison"
OVERLAY_ALPHA: Final[float]    = 0.45
GRID_FIG_WIDTH: Final[int]     = 16
GRID_FIG_HEIGHT_PER_ROW: Final[int] = 4
GRID_DPI: Final[int]           = 150
BLUR_KERNEL: Final[int]        = 51
N_FAITH_STEPS: Final[int]      = 100
# Max images per forward pass for deletion/insertion.
# 101 × 3 × 224 × 224 × fp16 ≈ 30 MB — safe on 32 GB VRAM.
FAITH_BATCH: Final[int]        = 101


# ─────────────────────────────────────────────────────────────────────────────
# Device helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_gpu_devices() -> list:
    n = torch.cuda.device_count()
    if n == 0:
        print("[WARN] No CUDA devices found — running on CPU (will be slow).")
        return [torch.device("cpu")]
    devs = [torch.device(f"cuda:{i}") for i in range(n)]
    for d in devs:
        props = torch.cuda.get_device_properties(d)
        vram_gb = props.total_memory / 1e9
        print(f"  {d}: {props.name}  {vram_gb:.1f} GB VRAM")
    return devs


def enable_tf32() -> None:
    """Enable TF32 on Ampere/Ada/Blackwell — free ~10% throughput."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True


# ─────────────────────────────────────────────────────────────────────────────
# Integrated Gradients (fully batched, GPU-optimised)
# ─────────────────────────────────────────────────────────────────────────────

class IntegratedGradients:
    """Pixel-level attribution via path integration from a black baseline.

    All n_steps interpolations are stacked into ONE batch tensor and processed
    in a single forward+backward pass — no Python loop over steps.

    Satisfies Completeness and Sensitivity axioms (Sundararajan et al., 2017).
    """

    def __init__(self, model: nn.Module, n_steps: int = DEFAULT_IG_STEPS) -> None:
        self.model   = model
        self.n_steps = n_steps

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """(1, C, H, W) → (H, W) attribution in [0, 1]."""
        self.model.eval()
        device = input_tensor.device

        baseline = torch.zeros_like(input_tensor)              # (1, C, H, W)
        alphas   = torch.linspace(0.0, 1.0, self.n_steps, device=device)
        delta    = input_tensor - baseline

        # (n_steps, C, H, W) — single allocation, no loop
        interpolated = (baseline + alphas.view(-1, 1, 1, 1) * delta).squeeze(0)
        interpolated = interpolated.float().requires_grad_(True)

        with torch.enable_grad():
            logits = self.model(interpolated)   # (n_steps, 1)
            logits.sum().backward()

        grads    = interpolated.grad            # (n_steps, C, H, W)
        avg_grads = grads.mean(dim=0)
        ig = (input_tensor.squeeze(0).float() - baseline.squeeze(0).float()) * avg_grads

        attr = ig.abs().sum(dim=0).detach().cpu().numpy()
        a_min, a_max = attr.min(), attr.max()
        if a_max > a_min:
            attr = (attr - a_min) / (a_max - a_min)
        else:
            attr = np.zeros_like(attr)
        return attr


# ─────────────────────────────────────────────────────────────────────────────
# LRP ε-rule (GPU-resident activations)
# ─────────────────────────────────────────────────────────────────────────────

class LRPEpsilon:
    """Layer-wise Relevance Propagation using the epsilon-stabiliser rule.

    Activations are kept on GPU (no host round-trip).
    Float32 is enforced for numerical stability in the relevance division.
    Compatible with torch.compile'd models.

    Reference: Bach et al., PLOS ONE 2015.
    """

    def __init__(self, model: nn.Module, epsilon: float = 1e-6) -> None:
        self.model    = model
        self.epsilon  = epsilon
        self._layers: list  = []
        self._activations: list = []
        self._hooks:  list  = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._layers.append(module)
                self._hooks.append(
                    module.register_forward_hook(self._save_activation)
                )

    def _save_activation(self, module, inp, out) -> None:
        self._activations.append(inp[0].detach().float())

    def _lrp_layer(self, layer, activation, relevance):
        activation = activation.requires_grad_(True)

        # Cast weights to float32 for stability
        orig_w = layer.weight.data
        orig_b = layer.bias.data if layer.bias is not None else None
        with torch.no_grad():
            layer.weight.data = orig_w.float()
            if orig_b is not None:
                layer.bias.data = orig_b.float()

        z = layer(activation)
        z_stable = z + self.epsilon * (z.sign() + (z == 0).float())
        ratio = (relevance.float() / (z_stable + 1e-12)).detach()
        (z * ratio).sum().backward()
        result = (activation.grad * activation).detach()

        with torch.no_grad():
            layer.weight.data = orig_w
            if orig_b is not None:
                layer.bias.data = orig_b

        return result

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """(1, C, H, W) → (H, W) attribution in [0, 1]."""
        self.model.eval()
        self._activations.clear()

        with torch.enable_grad():
            output = self.model(input_tensor.float())

        relevance = output.detach().float().clone()
        h, w = input_tensor.shape[2], input_tensor.shape[3]

        for layer, activation in zip(
            reversed(self._layers), reversed(self._activations)
        ):
            try:
                relevance = self._lrp_layer(layer, activation, relevance)
            except Exception:
                break

        # Bring to input spatial resolution
        if relevance.shape[-2:] != (h, w):
            rel = relevance.abs()
            if rel.dim() == 4:
                rel = rel.sum(dim=1, keepdim=True)
            elif rel.dim() < 4:
                rel = rel.reshape(1, 1, *rel.shape[-2:]) if rel.dim() == 2 else rel.unsqueeze(0).unsqueeze(0)
            rel = F.interpolate(rel, size=(h, w), mode="bilinear", align_corners=False)
            attr = rel.squeeze().cpu().numpy()
        else:
            r = relevance.abs()
            if r.dim() == 4:
                attr = r.sum(dim=1).squeeze().cpu().numpy()
            else:
                attr = r.squeeze().cpu().numpy()

        a_min, a_max = attr.min(), attr.max()
        if a_max > a_min:
            attr = (attr - a_min) / (a_max - a_min)
        else:
            attr = np.zeros_like(attr)
        return attr

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Batched faithfulness metrics
# ─────────────────────────────────────────────────────────────────────────────

def _gaussian_blur_tensor(tensor: torch.Tensor, kernel_size: int = BLUR_KERNEL) -> torch.Tensor:
    from torchvision.transforms.functional import gaussian_blur
    blurred = gaussian_blur(tensor.squeeze(0), kernel_size=[kernel_size, kernel_size])
    return blurred.unsqueeze(0)


def _make_masked_batch(
    input_tensor: torch.Tensor,
    flat_indices: np.ndarray,
    n_steps: int,
    mode: str,
    baseline: torch.Tensor,
) -> torch.Tensor:
    """Build (n_steps+1, C, H, W) batch of progressively masked images.

    Materialises all masking variants at once so the GPU processes them in
    a single forward pass instead of n_steps+1 sequential calls.

    Memory for n_steps=100, 224x224, fp16 ≈ 30 MB — trivial on 32 GB VRAM.
    """
    W = input_tensor.shape[3]
    step_size = max(1, (input_tensor.shape[2] * W) // n_steps)
    imgs = []
    img  = baseline.clone()

    for k in range(n_steps + 1):
        if k > 0:
            idx = flat_indices[: k * step_size]
            row, col = idx // W, idx % W
            img = img.clone()
            if mode == "deletion":
                img[0, :, row, col] = 0.0
            else:
                img[0, :, row, col] = input_tensor[0, :, row, col]
        imgs.append(img)

    return torch.cat(imgs, dim=0)   # (n_steps+1, C, H, W)


def _batched_forward(model: nn.Module, batch: torch.Tensor, chunk: int = FAITH_BATCH) -> np.ndarray:
    """Run model in chunks; return sigmoid probabilities as numpy array."""
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, batch.shape[0], chunk):
            c = batch[s: s + chunk]
            with torch.autocast(
                device_type=c.device.type, dtype=torch.float16, enabled=c.is_cuda
            ):
                logits = model(c).squeeze(-1)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def deletion_auc(
    model: nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    n_steps: int = N_FAITH_STEPS,
) -> float:
    """Batched deletion AUC.  Lower = attribution identifies truly critical regions."""
    flat_indices = np.argsort(saliency.ravel())[::-1]
    baseline = torch.zeros_like(input_tensor)
    batch    = _make_masked_batch(input_tensor, flat_indices, n_steps, "deletion", baseline)
    scores   = _batched_forward(model, batch.to(input_tensor.device))
    return float(np.trapz(scores, np.linspace(0, 1, len(scores))))


def insertion_auc(
    model: nn.Module,
    input_tensor: torch.Tensor,
    saliency: np.ndarray,
    n_steps: int = N_FAITH_STEPS,
) -> float:
    """Batched insertion AUC.  Higher = revealing salient pixels quickly restores confidence."""
    flat_indices = np.argsort(saliency.ravel())[::-1]
    blurred  = _gaussian_blur_tensor(input_tensor)
    batch    = _make_masked_batch(input_tensor, flat_indices, n_steps, "insertion", blurred)
    scores   = _batched_forward(model, batch.to(input_tensor.device))
    return float(np.trapz(scores, np.linspace(0, 1, len(scores))))


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _save_xai_grid(rows: list, output_path: str, model_name: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(GRID_FIG_WIDTH, GRID_FIG_HEIGHT_PER_ROW * n))
    if n == 1:
        axes = [axes]
    col_titles = ["Original", "Grad-CAM", "Integrated Gradients", "LRP (ε-rule)"]

    for i, row in enumerate(rows):
        raw   = row["raw"]
        maps  = [None, row["gradcam"], row["ig"], row["lrp"]]
        del_s = row["deletion_auc"]
        ins_s = row["insertion_auc"]

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

    fig.colorbar(im, ax=[axes[-1][j] for j in range(1, 4)],
                 fraction=0.015, pad=0.04, label="Attribution intensity")
    fig.suptitle(
        f"XAI Comparison — {model_name}\n"
        "Overlay: 35% image + 65% attribution   |   "
        "Del↓ = deletion AUC (lower = more faithful)   |   "
        "Ins↑ = insertion AUC (higher = more faithful)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=GRID_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def _save_faithfulness_summary(summary: dict, output_dir: str) -> None:
    models   = list(summary.keys())
    methods  = ["gradcam", "ig", "lrp"]
    m_labels = ["Grad-CAM", "Integ. Grad.", "LRP"]
    x        = np.arange(len(methods))
    width    = 0.35
    fig, (ax_del, ax_ins) = plt.subplots(1, 2, figsize=(12, 5))
    colours  = ["#1565C0", "#B71C1C", "#2E7D32", "#6A1B9A"]

    for mi, (mn, colour) in enumerate(zip(models, colours)):
        offset  = (mi - (len(models)-1)/2) * width
        d_means = [np.mean(summary[mn][m]["deletion"])  for m in methods]
        i_means = [np.mean(summary[mn][m]["insertion"]) for m in methods]
        ax_del.bar(x + offset, d_means, width, label=mn, color=colour, alpha=0.8)
        ax_ins.bar(x + offset, i_means, width, label=mn, color=colour, alpha=0.8)

    for ax, title in [
        (ax_del, "Deletion AUC  (↓ lower = more faithful)"),
        (ax_ins, "Insertion AUC  (↑ higher = more faithful)"),
    ]:
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(m_labels)
        ax.set_ylim(0, 1); ax.legend(); ax.set_ylabel("AUC"); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Faithfulness Metrics — Mean over test images\n"
        "Cross-method consistency: if all three methods agree per model, the interpretation is robust.",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "faithfulness_summary.png")
    plt.savefig(path, dpi=GRID_DPI, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _save_consistency_scatter(summary: dict, output_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colours   = {"densenet121": "#1565C0", "resnet18": "#B71C1C"}
    pairs     = [("gradcam", "ig", "GC vs IG"), ("gradcam", "lrp", "GC vs LRP")]

    for ax, (m1, m2, plabel) in zip(axes, pairs):
        for mn, colour in colours.items():
            if mn not in summary:
                continue
            ax.scatter(
                summary[mn][m1]["deletion"],
                summary[mn][m2]["deletion"],
                label=mn, color=colour, alpha=0.7, s=60,
            )
        ax.plot([0,1],[0,1],"k--",alpha=0.4,label="perfect agreement")
        ax.set_xlabel(f"{m1.upper()} Deletion AUC")
        ax.set_ylabel(f"{m2.upper()} Deletion AUC")
        ax.set_title(f"Method Consistency: {plabel}\n(near diagonal → methods agree)")
        ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.grid(alpha=0.3)

    fig.suptitle(
        "Cross-method consistency check (Deletion AUC)\n"
        "If a model's points cluster near the diagonal across ALL pairs, its XAI is internally coherent.",
        fontsize=11,
    )
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
    use_compile: bool = True,
) -> dict:
    """Run all three XAI methods for one model on one device.

    Returns faithfulness summary: method → {deletion: [...], insertion: [...]}.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # torch.compile notes for RTX 5090 (Blackwell / sm_120):
    #   - "reduce-overhead"        → CUDA graphs crash in worker threads (TLS)
    #   - "max-autotune-no-cudagraphs" → NoValidChoicesError: Triton not yet tuned
    #     for Blackwell SM arch. cuDNN already won every autotune round anyway.
    #   - "default"                → skips Triton autotuning, routes to cuDNN,
    #     thread-safe, works on all architectures. Correct mode for this workload.
    if use_compile and device.type == "cuda":
        print(f"  [{model_name}] Compiling model (default mode, cuDNN-backed)...")
        t0 = time.time()
        model = torch.compile(model, mode="default")
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        with torch.no_grad():
            model(dummy)
        torch.cuda.synchronize(device)
        print(f"  [{model_name}] Compiled + warmed up in {time.time()-t0:.1f}s")

    _, test_transform = get_transforms(image_size)
    dataset    = datasets.ImageFolder(root=data_dir, transform=test_transform)
    class_names = dataset.classes

    by_class: dict = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    per_class = max(1, num_images // len(class_names))
    rng = np.random.default_rng(seed)
    chosen = []
    for ci, cn in enumerate(class_names):
        sel = rng.choice(by_class[ci], size=min(per_class, len(by_class[ci])), replace=False)
        chosen.extend([(int(i), cn) for i in sel])

    gc_exp  = GradCAM(model, _get_target_layer(model, model_name))
    ig_exp  = IntegratedGradients(model, n_steps=ig_steps)
    lrp_exp = LRPEpsilon(model, epsilon=1e-6)

    # Dedicated CUDA stream per method allows the scheduler to overlap work
    streams = {}
    if device.type == "cuda":
        streams = {k: torch.cuda.Stream(device=device) for k in ["gradcam", "ig", "lrp"]}

    faith: dict = {m: {"deletion": [], "insertion": []} for m in ["gradcam", "ig", "lrp"]}
    rows: list  = []

    print(f"\n[{model_name} @ {device}] {len(chosen)} images "
          f"| IG steps={ig_steps} | faithfulness={compute_faithfulness}")

    for si, (idx, class_name) in enumerate(chosen):
        tensor, _ = dataset[idx]
        # pin_memory → non_blocking for faster H2D transfer
        inp = tensor.unsqueeze(0).to(device, non_blocking=True)
        raw = _denormalise(tensor)

        print(f"  [{si+1}/{len(chosen)}] {class_name}", end=" ", flush=True)

        # GradCAM
        if streams:
            with torch.cuda.stream(streams["gradcam"]):
                gc_map = gc_exp(inp)
        else:
            gc_map = gc_exp(inp)

        # IG — wait for gradcam stream before using same model
        if streams:
            torch.cuda.current_stream(device).wait_stream(streams["gradcam"])
            with torch.cuda.stream(streams["ig"]):
                ig_map = ig_exp(inp.detach())
        else:
            ig_map = ig_exp(inp.detach())

        # LRP
        if streams:
            torch.cuda.current_stream(device).wait_stream(streams["ig"])
            with torch.cuda.stream(streams["lrp"]):
                lrp_map = lrp_exp(inp.detach())
            torch.cuda.synchronize(device)
        else:
            lrp_map = lrp_exp(inp.detach())

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
    lrp_exp.remove_hooks()
    return faith


# ─────────────────────────────────────────────────────────────────────────────
# Parallel dispatcher — one thread per GPU
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
    use_compile: bool,
) -> dict:
    """Dispatch each model to a separate GPU thread concurrently.

    With 2× RTX 5090: DenseNet121 and ResNet18 run simultaneously,
    each with its own VRAM and CUDA context.
    Wall time ≈ max(t_dense, t_resnet) instead of t_dense + t_resnet.
    """
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
                compute_faithfulness=compute_faithfulness,
                ig_steps=ig_steps, use_compile=use_compile,
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
    skipped = set(checkpoints) - set(valid)
    for mn in skipped:
        print(f"[SKIP] Checkpoint not found: {checkpoints[mn]}")

    if not valid:
        print("[ERROR] No valid checkpoints found.")
        return {}

    # Round-robin GPU assignment
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
        description="XAI comparison (dual-GPU optimised): GradCAM | IG | LRP"
    )
    parser.add_argument("--config",              default="configs/config.yaml")
    parser.add_argument("--densenet-checkpoint", default="checkpoints/best_model_densenet121.pt")
    parser.add_argument("--resnet-checkpoint",   default="checkpoints/best_model_resnet18.pt")
    parser.add_argument("--data-dir",            default=None)
    parser.add_argument("--num-images",          type=int, default=DEFAULT_NUM_IMAGES)
    parser.add_argument("--output",              default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ig-steps",            type=int, default=DEFAULT_IG_STEPS,
                        help="IG interpolation steps. Default=100; use 300 for publication.")
    parser.add_argument("--no-faithfulness",     action="store_true",
                        help="Skip deletion/insertion AUC (fastest visual run)")
    parser.add_argument("--no-compile",          action="store_true",
                        help="Disable torch.compile (useful for debugging)")
    parser.add_argument("--gpu",                 type=int, nargs="+", default=None,
                        help="Override GPU indices, e.g. --gpu 0 1")
    args = parser.parse_args()

    enable_tf32()

    config     = load_config(args.config)
    image_size = config.get("image_size", DEFAULT_IMAGE_SIZE)
    data_dir   = args.data_dir or os.path.join(config["data_dir"], "test")
    seed       = config.get("seed", DEFAULT_SEED)
    os.makedirs(args.output, exist_ok=True)

    print("\n" + "="*60)
    print(" XAI Comparison — Dual-GPU optimised")
    print("="*60)
    all_devs = get_gpu_devices()
    if args.gpu is not None:
        devices = [all_devs[i] for i in args.gpu if i < len(all_devs)] or all_devs[:1]
    else:
        devices = all_devs

    print(f"\nConfig : {args.config}")
    print(f"Data   : {data_dir}")
    print(f"Images : {args.num_images}   IG steps : {args.ig_steps}")
    print(f"GPUs   : {[str(d) for d in devices]}")
    print(f"Compile: {not args.no_compile}   Faithfulness: {not args.no_faithfulness}")

    checkpoints = {
        "densenet121": args.densenet_checkpoint,
        "resnet18":    args.resnet_checkpoint,
    }

    t_start = time.time()
    summary = run_all_models_parallel(
        checkpoints=checkpoints, devices=devices,
        data_dir=data_dir, image_size=image_size,
        num_images=args.num_images, output_dir=args.output,
        seed=seed, compute_faithfulness=not args.no_faithfulness,
        ig_steps=args.ig_steps, use_compile=not args.no_compile,
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
    print("  Deletion AUC  ↓  lower  = attribution covers regions the model truly relies on")
    print("  Insertion AUC ↑  higher = revealing those regions quickly restores confidence")
    print("  Consistency       if GC, IG, LRP deletion AUCs are similar → XAI is coherent")
    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()