"""Grad-CAM: Gradient-weighted Class Activation Mapping.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data import IMAGENET_MEAN, IMAGENET_STD


def get_last_conv_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the last convolutional feature layer for each architecture."""
    name = model_name.lower()
    if name == "densenet121":
        return model.features.denseblock4
    elif name == "resnet18":
        return model.layer4[-1]
    elif name == "efficientnet_b0":
        return model.features[-1]
    elif name == "mobilenet_v2":
        return model.features[-1]
    else:
        raise ValueError(f"No Grad-CAM target layer defined for model: {model_name}")


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations = None
        self._gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single image tensor (1, C, H, W).

        Returns: (H, W) array with values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)
        logit = self.model(input_tensor).squeeze()
        self.model.zero_grad()
        logit.backward()

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def compute_gradcam(
    model: nn.Module, model_name: str, input_tensor: torch.Tensor
) -> np.ndarray:
    """Compute Grad-CAM heatmap for a single image tensor (1, C, H, W)."""
    target_layer = get_last_conv_layer(model, model_name)
    gcam = GradCAM(model, target_layer)
    heatmap = gcam(input_tensor)
    gcam.remove_hooks()
    return heatmap


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert ImageNet-normalized image tensor to uint8 numpy (H, W, 3)."""
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def overlay_heatmap(
    image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on raw image. Returns uint8 (H, W, 3)."""
    colormap = plt.colormaps["jet"]
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (alpha * heatmap_rgb + (1 - alpha) * image_np).astype(np.uint8)


def save_gradcam(heatmap: np.ndarray, image_np: np.ndarray, save_path: str) -> None:
    """Save original, Grad-CAM heatmap, and overlay side by side."""
    overlay = overlay_heatmap(image_np, heatmap)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")
    im = axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")
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
