import torch.nn as nn
from torchvision import models


def get_model(freeze_backbone: bool = False) -> nn.Module:
    """Create a DenseNet121 model for binary classification.

    Args:
        freeze_backbone: Whether to freeze the feature extraction layers.

    Returns:
        nn.Module with single logit output.
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier = nn.Linear(model.classifier.in_features, 1)
    return model
