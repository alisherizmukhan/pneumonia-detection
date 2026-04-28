"""Occlusion Sensitivity: sliding window perturbation analysis.

Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks",
ECCV 2014.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def compute_occlusion(
    model: nn.Module,
    input_tensor: torch.Tensor,
    window_size: int = 50,
    stride: int = 10,
) -> np.ndarray:
    """Sliding-window occlusion sensitivity map.

    For each (window_size × window_size) patch, zero it out and measure
    the drop in pneumonia probability. High drop = that region matters.

    Args:
        model: Trained model with single-logit binary output.
        input_tensor: (1, C, H, W) image tensor.
        window_size: Occlusion patch size in pixels.
        stride: Sliding step in pixels.

    Returns:
        (H, W) sensitivity map normalized to [0, 1].
    """
    model.eval()
    _, C, H, W = input_tensor.shape

    with torch.no_grad():
        baseline_prob = torch.sigmoid(model(input_tensor).squeeze()).item()

    sensitivity = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - window_size + 1, stride):
            for x in range(0, W - window_size + 1, stride):
                occluded = input_tensor.clone()
                occluded[:, :, y:y + window_size, x:x + window_size] = 0.0
                prob = torch.sigmoid(model(occluded).squeeze()).item()
                drop = max(0.0, baseline_prob - prob)
                sensitivity[y:y + window_size, x:x + window_size] += drop
                counts[y:y + window_size, x:x + window_size] += 1.0

    # Average over overlapping windows; leave uncovered border pixels at 0
    mask = counts > 0
    sensitivity[mask] /= counts[mask]

    s_min, s_max = sensitivity.min(), sensitivity.max()
    if s_max > s_min:
        sensitivity = (sensitivity - s_min) / (s_max - s_min)
    else:
        sensitivity = np.zeros_like(sensitivity)

    return sensitivity


def save_occlusion(
    sensitivity: np.ndarray, image_np: np.ndarray, save_path: str
) -> None:
    """Save occlusion sensitivity map, colorized map, and overlay."""
    colormap = plt.colormaps["plasma"]
    sens_rgb = (colormap(sensitivity)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.45 * sens_rgb + 0.55 * image_np).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    im = axes[1].imshow(sensitivity, cmap="plasma", vmin=0, vmax=1)
    axes[1].set_title("Occlusion Map")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
