"""Run all interpretability methods (Grad-CAM, LRP, Occlusion) for every model
on the fixed evaluation subset.

Output structure:
  results/interpretability/{model}/{image_name}/
    gradcam.png, gradcam.npy
    lrp.png,     lrp.npy
    occlusion.png, occlusion.npy
    metadata.json
"""
import matplotlib
matplotlib.use('Agg')
import os
import sys
import json
import argparse

import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from models import get_model
from utils import load_model, get_device, load_config, get_logger, set_seed
from data import get_transforms
from interpretability.gradcam import compute_gradcam, save_gradcam, denormalize
from interpretability.lrp import compute_lrp, save_lrp
from interpretability.occlusion import compute_occlusion, save_occlusion


def get_eval_subset(data_dir: str, n_per_class: int = 5) -> list:
    """Load n_per_class images from data/eval_subset/, falling back to test set."""
    eval_dir = os.path.join(os.path.dirname(data_dir), "eval_subset")
    base_dir = eval_dir if os.path.isdir(eval_dir) else os.path.join(data_dir, "test")

    subset = []
    for class_name, label in [("NORMAL", 0), ("PNEUMONIA", 1)]:
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = sorted([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:n_per_class]
        for fname in files:
            subset.append({
                "path": os.path.join(class_dir, fname),
                "label": label,
                "class": class_name,
                "name": os.path.splitext(fname)[0],
            })
    return subset


def run_interpretability(config: dict) -> None:
    logger = get_logger("interpretability")
    set_seed(config.get("seed", 42))
    device = get_device()
    logger.info(f"Device: {device}")

    models_list = config.get(
        "models", ["densenet121", "resnet18", "efficientnet_b0", "mobilenet_v2"]
    )
    checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
    results_dir = config.get("results_dir", "results")
    data_dir = config["data_dir"]
    image_size = config.get("image_size", 224)
    threshold = config.get("threshold", 0.3)

    subset = get_eval_subset(data_dir)
    if not subset:
        logger.error(
            "No evaluation images found. Populate data/eval_subset/ or "
            "ensure data/chest_xray/test/ exists."
        )
        return

    logger.info(f"Evaluation subset: {len(subset)} images")
    _, test_transform = get_transforms(image_size)

    for model_name in models_list:
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}.pt")
        if not os.path.exists(ckpt_path):
            logger.warning(f"Checkpoint not found: {ckpt_path} — skipping {model_name}")
            continue

        logger.info(f"Running interpretability for {model_name}...")
        model = get_model(model_name)
        model = load_model(model, ckpt_path, device)
        model = model.to(device)
        model.eval()

        for img_info in subset:
            img_path = img_info["path"]
            img_name = img_info["name"]
            true_class = img_info["class"]

            out_dir = os.path.join(results_dir, "interpretability", model_name, img_name)
            os.makedirs(out_dir, exist_ok=True)

            pil_img = Image.open(img_path).convert("RGB")
            tensor = test_transform(pil_img).unsqueeze(0).to(device)
            image_np = denormalize(tensor.squeeze(0).cpu())

            with torch.no_grad():
                prob = torch.sigmoid(model(tensor).squeeze()).item()
            pred_class = "PNEUMONIA" if prob >= threshold else "NORMAL"

            # --- Grad-CAM ---
            gradcam_map = None
            try:
                gradcam_map = compute_gradcam(model, model_name, tensor)
                save_gradcam(gradcam_map, image_np, os.path.join(out_dir, "gradcam.png"))
                np.save(os.path.join(out_dir, "gradcam.npy"), gradcam_map)
            except Exception as exc:
                logger.warning(f"Grad-CAM failed [{model_name}/{img_name}]: {exc}")

            # --- LRP ---
            lrp_map = None
            try:
                lrp_map = compute_lrp(model, tensor)
                save_lrp(lrp_map, image_np, os.path.join(out_dir, "lrp.png"))
                np.save(os.path.join(out_dir, "lrp.npy"), lrp_map)
            except Exception as exc:
                logger.warning(f"LRP failed [{model_name}/{img_name}]: {exc}")

            # --- Occlusion ---
            occlusion_map = None
            try:
                occlusion_map = compute_occlusion(model, tensor)
                save_occlusion(occlusion_map, image_np, os.path.join(out_dir, "occlusion.png"))
                np.save(os.path.join(out_dir, "occlusion.npy"), occlusion_map)
            except Exception as exc:
                logger.warning(f"Occlusion failed [{model_name}/{img_name}]: {exc}")

            meta = {
                "model": model_name,
                "image": img_name,
                "true_class": true_class,
                "predicted_class": pred_class,
                "probability": round(float(prob), 4),
                "threshold": threshold,
                "methods_computed": {
                    "gradcam": gradcam_map is not None,
                    "lrp": lrp_map is not None,
                    "occlusion": occlusion_map is not None,
                },
            }
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2)

            logger.info(
                f"  {model_name}/{img_name}: "
                f"prob={prob:.3f}  pred={pred_class}  true={true_class}"
            )

    logger.info("Interpretability pipeline complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Run interpretability pipeline (Grad-CAM, LRP, Occlusion) for all models"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_interpretability(config)


if __name__ == "__main__":
    main()
