
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configuration
TRAIN_DIR = "datasets/dog_emotions/train"
VAL_DIR = "datasets/dog_emotions/val"
SAVE_PATH = "models/efficientnet_b0_improved.pth"

EPOCHS = 50
BATCH_SIZE = 16
LR = 0.0001
PATIENCE = 10
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.05
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Validation without augmentation
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)

if train_ds.classes != val_ds.classes:
    raise ValueError("Training and validation classes do not match.")

classes = train_ds.classes
num_classes = len(classes)
print("Classes:", classes)
print("Training images:", len(train_ds))
print("Validation images:", len(val_ds))

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0,
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0,
    pin_memory=torch.cuda.is_available()
)

# Calculate class weights from training data
counts = np.bincount(train_ds.targets, minlength=num_classes)
if np.any(counts == 0):
    raise ValueError("A class has no training images.")

weights = len(train_ds) / (num_classes * counts)
class_weights = torch.tensor(
    weights, dtype=torch.float32, device=device
)
print("Training class counts:", counts.tolist())
print("Class weights:", weights.round(3).tolist())

# Load pretrained EfficientNet-B0
model = models.efficientnet_b0(
    weights=EfficientNet_B0_Weights.DEFAULT
)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=3
)

best_f1 = -1.0
best_accuracy = 0.0
best_state = None
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    train_loss = total_loss / len(train_ds)

    # Validation
    model.eval()
    y_true, y_pred = [], []
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    val_loss /= len(val_ds)
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    scheduler.step(val_f1)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}% | "
        f"Macro F1: {val_f1 * 100:.2f}% | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # Save best model based on validation macro F1
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_accuracy = val_acc
        best_state = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0
        print("  Best validation macro F1 improved.")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break

# Save best checkpoint
if best_state is None:
    raise RuntimeError("No model checkpoint was produced.")

Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_name": "efficientnet_b0_improved",
    "model_state_dict": best_state,
    "classes": classes,
    "best_val_accuracy": best_accuracy,
    "best_val_macro_f1": best_f1,
    "image_size": 224
}, SAVE_PATH)

print("\nTraining complete.")
print(f"Best validation accuracy: {best_accuracy * 100:.2f}%")
print(f"Best validation macro F1: {best_f1 * 100:.2f}%")
print("Model saved to:", SAVE_PATH)