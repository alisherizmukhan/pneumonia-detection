"""Generate architecture diagrams for all three models using torchviz.

Requirements:
    pip install torchviz graphviz

Output:
    results/architecture/baseline_cnn.png
    results/architecture/resnet18.png
    results/architecture/densenet121.png
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from torchviz import make_dot
from models import get_model

OUTPUT_DIR = "results/architecture"
IMAGE_SIZE = 224


def save_diagram(model_name: str) -> None:
    model = get_model(model_name, pretrained=False)
    model.eval()

    x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    out = model(x)

    dot = make_dot(
        out,
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )
    dot.attr(rankdir="TB", size="12,20")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, model_name)
    dot.render(path, format="png", cleanup=True)
    print(f"Saved: {path}.png")


if __name__ == "__main__":
    for name in ["baseline", "resnet18", "densenet121"]:
        print(f"Generating diagram for {name}...")
        save_diagram(name)
    print("Done.")
