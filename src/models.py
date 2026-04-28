import torch.nn as nn
from torchvision import models


def get_model(name: str) -> nn.Module:
    """Factory: return a pretrained model with a single binary output logit.

    Supported names: densenet121, resnet18, efficientnet_b0, mobilenet_v2
    """
    name = name.lower()

    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 1)

    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 1)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: "
            "densenet121, resnet18, efficientnet_b0, mobilenet_v2"
        )

    return model
