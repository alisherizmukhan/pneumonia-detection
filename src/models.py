import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """Simple 3-layer CNN for binary classification (trained from scratch)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet18Model(nn.Module):
    """Transfer learning model using pretrained ResNet-18."""

    def __init__(self, freeze_backbone: bool = False, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.backbone(x)


def _build_densenet121(freeze_backbone: bool = False, pretrained: bool = True) -> nn.Module:
    """Create a DenseNet-121 model for binary classification."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model


def get_model(model_name: str = "densenet121", freeze_backbone: bool = False, pretrained: bool = True) -> nn.Module:
    """Factory function to create a model by name.

    Args:
        model_name: 'baseline', 'resnet18', or 'densenet121'.
        freeze_backbone: Whether to freeze pretrained backbone layers.

    Returns:
        nn.Module with single logit output.
    """
    if model_name == "baseline":
        return BaselineCNN()
    elif model_name == "resnet18":
        return ResNet18Model(freeze_backbone=freeze_backbone, pretrained=pretrained)
    elif model_name == "densenet121":
        return _build_densenet121(freeze_backbone=freeze_backbone, pretrained=pretrained)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. Choose 'baseline', 'resnet18', or 'densenet121'."
        )
