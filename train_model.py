import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "dataset"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = [
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "leaf_miner",
    "spider_mites",
    "leaf_curl_virus",
    "cercospora_leaf_mold",
    "insect_damage",
    "healthy",
]


def get_data_loaders(batch_size: int = 32, img_size: int = 224):
    train_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transforms)

    if sorted(train_dataset.classes) != sorted(CLASS_NAMES):
        print("WARNING: Folder class names do not exactly match CLASS_NAMES.")
        print("Found classes:", train_dataset.classes)

    # num_workers=0 avoids Windows file locking issues.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset.classes


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model(
    num_epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    img_size: int = 224,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, val_loader, cls = get_data_loaders(batch_size=batch_size, img_size=img_size)
    num_classes = len(cls)
    print(f"Detected {num_classes} classes: {cls}")

    model = build_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_model_path = MODELS_DIR / "tomato_resnet18.pth"

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_corrects / val_total

        print(
            f"Epoch {epoch}/{num_epochs} "
            f"- Train loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f} "
            f"- Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": cls,
                    "img_size": img_size,
                },
                best_model_path,
            )
            print(f"Saved new best model to {best_model_path} (val_acc={best_val_acc:.4f})")

    print(f"Training complete. Best val acc: {best_val_acc:.4f}")
    print(f"Best model saved at: {best_model_path}")


if __name__ == "__main__":
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise SystemExit(
            f"Expected training and validation folders at:\n{TRAIN_DIR}\n{VAL_DIR}\n"
            "Please organize the dataset as described in README.md before training."
        )

    train_model()

