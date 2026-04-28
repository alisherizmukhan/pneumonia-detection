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
from typing import Final

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

from data import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from models import get_model
from utils import load_model, get_device, load_config

DEFAULT_SEED: Final[int] = 42
DEFAULT_MODEL: Final[str] = "densenet121"
DEFAULT_IMAGE_SIZE: Final[int] = 224
DEFAULT_NUM_IMAGES: Final[int] = 8
DEFAULT_OUTPUT_DIR: Final[str] = "results/gradcam"
OVERLAY_ALPHA: Final[float] = 0.45
GRID_FIG_WIDTH: Final[int] = 12
GRID_FIG_HEIGHT_PER_ROW: Final[int] = 4
GRID_DPI: Final[int] = 150
COLORBAR_FRACTION: Final[float] = 0.046
COLORBAR_PAD: Final[float] = 0.04

GradCAMRow = tuple[np.ndarray, np.ndarray, np.ndarray, str]


def _get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the last convolutional feature layer for each architecture."""
    if model_name == "densenet121":
        return model.features.denseblock4
    if model_name == "resnet18":
        return model.backbone.layer4[-1]
    if model_name == "baseline":
        return model.features[-3]  # last Conv2d before final ReLU+MaxPool
    raise ValueError(f"No target layer defined for model: {model_name}")


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(
        self,
        module: nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._activations = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute Grad-CAM heatmap for a single image tensor (1, C, H, W).

        Returns:
            heatmap: np.ndarray of shape (H, W) with values in [0, 1].
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logit: torch.Tensor = self.model(input_tensor).squeeze()
        self.model.zero_grad()
        logit.backward()

        weights: torch.Tensor = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam: torch.Tensor = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        h: int = input_tensor.shape[2]
        w: int = input_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam_np: np.ndarray = cam.squeeze().cpu().numpy()

        cam_min: float = cam_np.min()
        cam_max: float = cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def remove_hooks(self) -> None:
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = OVERLAY_ALPHA,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on a raw image.

    Args:
        image_np: (H, W, 3) uint8 array.
        heatmap:  (H, W) float array in [0, 1].
        alpha:    Heatmap opacity.

    Returns:
        (H, W, 3) uint8 blended image.
    """
    colormap = plt.colormaps["jet"]
    heatmap_rgb: np.ndarray = (colormap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    blended: np.ndarray = (alpha * heatmap_rgb + (1 - alpha) * image_np).astype(np.uint8)
    return blended


def _denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised image tensor to uint8 numpy (H, W, 3)."""
    mean: np.ndarray = np.array(IMAGENET_MEAN)
    std: np.ndarray = np.array(IMAGENET_STD)
    img: np.ndarray = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def generate_gradcam_grid(
    model: nn.Module,
    model_name: str,
    data_dir: str,
    image_size: int,
    num_images: int,
    output_dir: str,
    device: torch.device,
    seed: int = DEFAULT_SEED,
) -> None:
    """Sample images from data_dir, compute Grad-CAM, and save grid figures."""
    os.makedirs(output_dir, exist_ok=True)
    torch.manual_seed(seed)

    _, test_transform = get_transforms(image_size)
    dataset = datasets.ImageFolder(root=data_dir, transform=test_transform)
    class_names: list[str] = dataset.classes

    by_class: dict[int, list[int]] = {i: [] for i in range(len(class_names))}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class[label].append(idx)

    grad_cam = GradCAM(model, _get_target_layer(model, model_name))

    per_class: int = max(1, num_images // len(class_names))
    all_rows: list[GradCAMRow] = []

    for class_idx, class_name in enumerate(class_names):
        indices: list[int] = by_class[class_idx]
        np.random.seed(seed)
        chosen: np.ndarray = np.random.choice(
            indices, size=min(per_class, len(indices)), replace=False
        )

        rows: list[GradCAMRow] = []
        for idx in chosen:
            tensor, _ = dataset[idx]
            inp: torch.Tensor = tensor.unsqueeze(0).to(device)
            heatmap: np.ndarray = grad_cam(inp)
            raw: np.ndarray = _denormalise(tensor)
            overlay: np.ndarray = overlay_heatmap(raw, heatmap)
            rows.append((raw, heatmap, overlay, class_name))

        _save_class_grid(rows, class_name, output_dir, model_name)
        all_rows.extend(rows)

    _save_combined_grid(all_rows, output_dir, model_name)
    grad_cam.remove_hooks()


def _save_class_grid(
    rows: list[GradCAMRow],
    class_name: str,
    output_dir: str,
    model_name: str,
) -> None:
    n: int = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(GRID_FIG_WIDTH, GRID_FIG_HEIGHT_PER_ROW * n))
    if n == 1:
        axes = [axes]

    for i, (raw, heatmap, overlay, _) in enumerate(rows):
        axes[i][0].imshow(raw)
        axes[i][0].set_title("Original" if i == 0 else "")
        axes[i][0].axis("off")

        im = axes[i][1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[i][1].set_title("Grad-CAM" if i == 0 else "")
        axes[i][1].axis("off")

        axes[i][2].imshow(overlay)
        axes[i][2].set_title("Overlay" if i == 0 else "")
        axes[i][2].axis("off")

    fig.colorbar(im, ax=axes[-1][1], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
    fig.suptitle(f"Grad-CAM — {model_name} — {class_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    path: str = os.path.join(output_dir, f"gradcam_{model_name}_{class_name.lower()}.png")
    plt.savefig(path, dpi=GRID_DPI, bbox_inches="tight")
    plt.close()


def _save_combined_grid(
    rows: list[GradCAMRow],
    output_dir: str,
    model_name: str,
) -> None:
    n: int = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(GRID_FIG_WIDTH, GRID_FIG_HEIGHT_PER_ROW * n))
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

    fig.colorbar(im, ax=axes[-1][1], fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
    fig.suptitle(f"Grad-CAM — {model_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    path: str = os.path.join(output_dir, f"gradcam_{model_name}_combined.png")
    plt.savefig(path, dpi=GRID_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM grid: {path}")


def gradcam_for_pil_image(
    image: Image.Image,
    model: nn.Module,
    model_name: str,
    image_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Grad-CAM for a PIL image (used by the Streamlit app).

    Returns:
        (heatmap, overlay): spatial heatmap (H, W) and blended overlay (H, W, 3).
    """
    _, test_transform = get_transforms(image_size)
    tensor: torch.Tensor = test_transform(image).unsqueeze(0).to(device)

    grad_cam = GradCAM(model, _get_target_layer(model, model_name))
    heatmap: np.ndarray = grad_cam(tensor)
    grad_cam.remove_hooks()

    raw: np.ndarray = _denormalise(tensor.squeeze(0).cpu())
    overlay: np.ndarray = overlay_heatmap(raw, heatmap)
    return heatmap, overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualisations")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="checkpoints/best_model_densenet121.pt")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Image directory (defaults to config data_dir/test)")
    parser.add_argument("--num-images", type=int, default=DEFAULT_NUM_IMAGES,
                        help="Total images to visualise (split evenly across classes)")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    config: dict = load_config(args.config)
    model_name: str = config.get("model", DEFAULT_MODEL)
    image_size: int = config.get("image_size", DEFAULT_IMAGE_SIZE)
    data_dir: str = args.data_dir or os.path.join(config["data_dir"], "test")
    device: torch.device = get_device()

    model: nn.Module = get_model(model_name=model_name, freeze_backbone=False, pretrained=False)
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
        seed=config.get("seed", DEFAULT_SEED),
    )
    print("Done.")


if __name__ == "__main__":
    main()
