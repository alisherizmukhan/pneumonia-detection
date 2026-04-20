import argparse

import torch
from PIL import Image
import torchvision.transforms as transforms

from models import get_model
from utils import load_model, get_device, load_config
from data import IMAGENET_MEAN, IMAGENET_STD

DEFAULT_CONFIG = "configs/config.yaml"


def load_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def predict(image_path, model_path, model_name="densenet121", threshold=0.3):
    device = get_device()

    model = get_model(model_name)
    model = load_model(model, model_path, device)
    model = model.to(device)
    model.eval()

    image = load_image(image_path).to(device)

    with torch.no_grad():
        output = model(image).squeeze(1)
        prob = torch.sigmoid(output).item()

    label = "PNEUMONIA" if prob >= threshold else "NORMAL"

    print(f"Probability: {prob:.4f}")
    print(f"Threshold:   {threshold}")
    print(f"Prediction:  {label}")
    return label, prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a chest X-ray image")
    parser.add_argument("--image", type=str, required=True, help="Path to X-ray image")
    parser.add_argument("--model", type=str, default="checkpoints/best_model_densenet121.pt")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Config file for threshold")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold")
    args = parser.parse_args()

    config = load_config(args.config)
    threshold = args.threshold if args.threshold is not None else config.get("threshold", 0.3)
    model_name = config.get("model", "densenet121")
    predict(args.image, args.model, model_name=model_name, threshold=threshold)