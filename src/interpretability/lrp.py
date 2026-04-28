"""Layer-wise Relevance Propagation (LRP) via Captum.

Bach et al., "On Pixel-wise Explanations for Non-Linear Classifier Decisions
by Layer-wise Relevance Propagation", PLOS ONE 2015.

Requires: pip install captum
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class _BinaryOutputWrapper(nn.Module):
    """Ensure the model returns (batch, 1) for Captum compatibility."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if out.dim() == 1:
            return out.unsqueeze(1)
        return out


def compute_lrp(model: nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    """Compute LRP attribution map.

    Args:
        model: Trained nn.Module with single-logit binary output.
        input_tensor: (1, C, H, W) image tensor.

    Returns:
        (H, W) attribution map normalized to [0, 1].

    Raises:
        ImportError: If captum is not installed.
    """
    try:
        from captum.attr import LRP
    except ImportError:
        raise ImportError("captum is required for LRP: pip install captum")

    model.eval()
    wrapped = _BinaryOutputWrapper(model)
    inp = input_tensor.clone().requires_grad_(True)

    lrp = LRP(wrapped)
    # target=0 selects the single output class
    attribution = lrp.attribute(inp, target=0)

    attr_np = attribution.squeeze().detach().cpu().numpy()
    if attr_np.ndim == 3:
        # Sum over color channels to get a single spatial map
        attr_np = attr_np.sum(axis=0)

    attr_min, attr_max = attr_np.min(), attr_np.max()
    if attr_max > attr_min:
        attr_np = (attr_np - attr_min) / (attr_max - attr_min)
    else:
        attr_np = np.zeros_like(attr_np)

    return attr_np


def save_lrp(attribution: np.ndarray, image_np: np.ndarray, save_path: str) -> None:
    """Save LRP attribution map, colorized map, and overlay."""
    colormap = plt.colormaps["hot"]
    attr_rgb = (colormap(attribution)[:, :, :3] * 255).astype(np.uint8)
    overlay = (0.5 * attr_rgb + 0.5 * image_np).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    im = axes[1].imshow(attribution, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("LRP Map")
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
