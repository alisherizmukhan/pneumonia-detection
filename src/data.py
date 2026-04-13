from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import torch


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(image_size: int = 224):
    """Return train and test transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, test_transform


def compute_class_weights(data_dir: str):
    """Compute class weights based on training set distribution.

    Returns:
        torch.Tensor with weight for the positive class (for BCEWithLogitsLoss pos_weight).
    """
    train_dir = os.path.join(data_dir, "train")
    dataset = datasets.ImageFolder(root=train_dir)
    targets = [t for _, t in dataset.samples]
    class_counts = [0, 0]
    for t in targets:
        class_counts[t] += 1
    # pos_weight = num_negative / num_positive
    if class_counts[1] > 0:
        pos_weight = class_counts[0] / class_counts[1]
    else:
        pos_weight = 1.0
    return torch.tensor([pos_weight], dtype=torch.float32), class_counts


def get_dataloaders(data_dir: str, batch_size: int = 32, image_size: int = 224,
                    num_workers: int = 4):
    """Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to root data directory containing train/, val/, test/.
        batch_size: Batch size for dataloaders.
        image_size: Image resize dimension.
        num_workers: Number of dataloader workers.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_transform, test_transform = get_transforms(image_size)

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=test_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
