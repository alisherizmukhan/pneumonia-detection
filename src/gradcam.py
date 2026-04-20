"""Grad-CAM visualisation for pneumonia detection models.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.

Usage:
    python src/gradcam.py --config configs/config.yaml \
                          --model checkpoints/best_model_densenet121.pt \
                          --data-dir data/chest_xray/test \
                          --num-images 8 \
                          --output results/gradcam
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import datasets, transforms

from data import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from models import get_model, BaselineCNN
from utils import load_model, get_device, load_config


def _get_target_layer(model, model_name: str):
    """Return the last convolutional feature layer for each architecture."""
    if model_name == "densenet121":
        return model.features.denseblock4
    if model_name == "resnet18":
        return model.backbone.layer4[-1]
    if model_name == "baseline":
        # Last Conv2d in the features Sequential
        return model.features[-3]  # index of last Conv2d before ReLU+MaxPool
    raise ValueError(f"No target layer defined for model: {model_name}")


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single image tensor (1, C, H, W).

        Returns:
            heatmap: np.ndarray of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logit = self.model(input_tensor).squeeze()
        self.model.zero_grad()
        logit.backward()

        # Global average pool the gradients over spatial dims
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Upsample to input spatial size
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on a raw image.

    Args:
        image_np: (H, W, 3) uint8 array.
        heatmap:  (H, W) float array in [0, 1].
        alpha:    Heatmap opacity.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    colormap = plt.colormaps["jet"]
    heatmap_rgb = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended = (alpha * heatmap_rgb + (1 - alpha) * image_np).astype(np.uint8)
    return blended


def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor to uint8 numpy (H, W, 3)."""
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def generate_gradcam_grid(
    model: torch.nn.Module,
    model_name: str,
    data_dir: str,
    image_size: int,
    num_images: int,
    output_dir: str,
    device: torch.device,
    seed: int = 42,
):
    """Sample images from data_dir, compute Grad-CAM, and save a grid figure.

    Saves one grid per class (NORMAL / PNEUMONIA) and a combined grid.
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    _, test_transform = get_transforms(image_size)
    dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)
    class_names = dataset.classes  # typically ['NORMAL', 'PNEUMONIA']

    # Separate indices by class
    by_class: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    grad_cam = GradCAM(model, _get_target_layer(model, model_name))

    per_class = max(1, num_images // len(class_names))
    all_rows = []

    for class_idx, class_name in enumerate(class_names):
        indices = by_class[class_idx]
        np.random.seed(seed)
        chosen = np.random.choice(indices, size=min(per_class, len(indices)), replace=False)

        rows = []
        for idx in chosen:
            tensor, label = dataset[idx]
            inp = tensor.unsqueeze(0).to(device)
            heatmap = grad_cam(inp)
            raw = _denormalise(tensor)
            overlay = overlay_heatmap(raw, heatmap)
            rows.append((raw, heatmap, overlay, class_name))

        _save_class_grid(rows, class_name, output_dir, model_name)
        all_rows.extend(rows)

    _save_combined_grid(all_rows, output_dir, model_name)
    grad_cam.remove_hooks()


def _save_class_grid(rows, class_name, output_dir, model_name):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (raw, heatmap, overlay, label) in enumerate(rows):
        axes[i][0].imshow(raw)
        axes[i][0].set_title("Original" if i == 0 else "")
        axes[i][0].axis("off")

        im = axes[i][1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[i][1].set_title("Grad-CAM" if i == 0 else "")
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("Overlay" if i == 0 else "")
        axes[i][2].axis("off")

    fig.colorbar(im, ax=axes[-1][1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Grad-CAM — {model_name} — {class_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"gradcam_{model_name}_{class_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_combined_grid(rows, output_dir, model_name):
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = [axes]

    for i, (raw, heatmap, overlay, label) in enumerate(rows):
        axes[i][0].imshow(raw)
        axes[i][0].set_ylabel(label, rotation=0, labelpad=60, fontsize=9)
        axes[i][0].set_title("Original" if i == 0 else "")
        axes[i][0].axis("off")

        im = axes[i][1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[i][1].set_title("Grad-CAM" if i == 0 else "")
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("Overlay" if i == 0 else "")
        axes[i][2].axis("off")

    fig.colorbar(im, ax=axes[-1][1], fraction=0.046, pad=0.04)
    fig.suptitle(f"Grad-CAM — {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, f"gradcam_{model_name}_combined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM grid: {path}")


def gradcam_for_pil_image(
    image: Image.Image,
    model: torch.nn.Module,
    model_name: str,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM for a PIL image (used by the Streamlit app).

    Returns:
        (heatmap, overlay) as uint8 numpy arrays (H, W) and (H, W, 3).
    """
    _, test_transform = get_transforms(image_size)
    tensor = test_transform(image).unsqueeze(0).to(device)

    grad_cam = GradCAM(model, _get_target_layer(model, model_name))
    heatmap = grad_cam(tensor)
    grad_cam.remove_hooks()

    raw = _denormalise(tensor.squeeze(0).cpu())
    overlay = overlay_heatmap(raw, heatmap)
    return heatmap, overlay


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualisations")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="checkpoints/best_model_densenet121.pt")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Image directory (defaults to config data_dir/test)")
    parser.add_argument("--num-images", type=int, default=8,
                        help="Total images to visualise (split evenly across classes)")
    parser.add_argument("--output", type=str, default="results/gradcam")
    args = parser.parse_args()

    config = load_config(args.config)
    model_name = config.get("model", "densenet121")
    image_size = config.get("image_size", 224)
    data_dir = args.data_dir or os.path.join(config["data_dir"], "test")
    device = get_device()

    model = get_model(model_name=model_name, freeze_backbone=False, pretrained=False)
    model = load_model(model, args.model, device)
    model = model.to(device)

    print(f"Generating Grad-CAM for {model_name} on {data_dir}")
    generate_gradcam_grid(
        model=model,
        model_name=model_name,
        data_dir=data_dir,
        image_size=image_size,
        num_images=args.num_images,
        output_dir=args.output,
        device=device,
        seed=config.get("seed", 42),
    )
    print("Done.")


if __name__ == "__main__":
    main()
