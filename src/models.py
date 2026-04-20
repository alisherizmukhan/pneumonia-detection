import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Simple 3-layer CNN trained from scratch — serves as the non-transfer-learning baseline."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(model_name: str = "densenet121", freeze_backbone: bool = False) -> nn.Module:
    """Return model by name.

    Args:
        model_name: One of 'baseline', 'resnet18', 'densenet121'.
        freeze_backbone: If True, freeze all layers except the final classifier.

    Returns:
        nn.Module with single logit output.
    """
    if model_name == "baseline":
        return BaselineCNN()

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model

    if model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        return model

    raise ValueError(
        f"Unknown model: {model_name!r}. Choose from: baseline, resnet18, densenet121"
    )
