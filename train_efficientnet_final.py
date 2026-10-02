import os
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = r"C:\Users\acer\dogs-emotion-recognition"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "dog_emotions_clean",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "datasets",
    "dog_emotions_clean",
    "val"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "efficientnet_b0_final80.pth"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# Training settings
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30

# Main changes
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# Early stopping
PATIENCE = 7

# DataLoader
NUM_WORKERS = 0       # Safe for Windows
PIN_MEMORY = True


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("FINAL EFFICIENTNET-B0 TRAINING")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


# ============================================================
# 3. CHECK DATASET PATHS
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR}"
    )


# ============================================================
# 4. IMAGE TRANSFORMS
# ============================================================

# Training:
# Moderate augmentation to improve generalization.
#
# IMPORTANT:
# Validation uses the SAME final image size (224x224)
# without random augmentation.

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# 5. LOAD DATASETS
# ============================================================

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

classes = train_dataset.classes
num_classes = len(classes)

print("Dataset information")
print("-" * 70)

print(f"Train images      : {len(train_dataset)}")
print(f"Validation images : {len(val_dataset)}")
print(f"Classes           : {classes}")
print(f"Number of classes : {num_classes}")

print()


# ============================================================
# 6. CHECK CLASS CONSISTENCY
# ============================================================

if train_dataset.classes != val_dataset.classes:
    raise ValueError(
        "Training and validation classes do not match!\n"
        f"Train: {train_dataset.classes}\n"
        f"Val  : {val_dataset.classes}"
    )


# ============================================================
# 7. CLASS DISTRIBUTION
# ============================================================

train_targets = np.array(train_dataset.targets)

class_counts = np.bincount(
    train_targets,
    minlength=num_classes
)

print("Training class distribution")
print("-" * 70)

for class_name, count in zip(classes, class_counts):
    print(f"{class_name:10s}: {count}")

print()


# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

# Balanced class weighting:
#
# weight = total_samples /
#          (number_of_classes * samples_in_class)

class_weights = len(train_dataset) / (
    num_classes * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)

print("Class weights")
print("-" * 70)

for class_name, weight in zip(
    classes,
    class_weights.cpu().numpy()
):
    print(f"{class_name:10s}: {weight:.4f}")

print()


# ============================================================
# 9. DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# ============================================================
# 10. LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("Loading pretrained EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)


# Replace final classifier
in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(device)

print("EfficientNet-B0 loaded.")
print()


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# 12. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 13. LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 14. TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# 15. VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    total = len(all_labels)

    val_loss = running_loss / total

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    val_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return (
        val_loss,
        val_accuracy,
        val_macro_f1
    )


# ============================================================
# 16. TRAINING LOOP
# ============================================================

print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

print(f"Learning rate   : {LEARNING_RATE}")
print(f"Weight decay    : {WEIGHT_DECAY}")
print(f"Label smoothing : {LABEL_SMOOTHING}")
print(f"Batch size      : {BATCH_SIZE}")
print(f"Epochs          : {NUM_EPOCHS}")
print(f"Early stopping  : {PATIENCE} epochs")
print()


best_val_f1 = 0.0
best_val_accuracy = 0.0
best_epoch = 0

best_model_state = None

epochs_without_improvement = 0

start_time = time.time()


for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    val_loss, val_accuracy, val_macro_f1 = validate(
        model,
        val_loader,
        criterion,
        device
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler.step(val_macro_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = time.time() - epoch_start

    # --------------------------------------------------------
    # Check improvement
    # --------------------------------------------------------

    if val_macro_f1 > best_val_f1:

        best_val_f1 = val_macro_f1
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        # Save checkpoint immediately
        torch.save(
            {
                "model_state_dict": best_model_state,
                "classes": classes,
                "num_classes": num_classes,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy,
                "best_val_macro_f1": best_val_f1,
                "best_epoch": best_epoch
            },
            MODEL_PATH
        )

        epochs_without_improvement = 0

        marker = "<-- BEST"

    else:

        epochs_without_improvement += 1

        marker = ""

    # --------------------------------------------------------
    # Print epoch results
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy*100:.2f}% | "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy*100:.2f}% "
        f"Val Macro F1: {val_macro_f1*100:.2f}% "
        f"| LR: {current_lr:.2e} "
        f"| Time: {epoch_time:.1f}s "
        f"{marker}"
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print()
        print(
            f"Early stopping triggered after "
            f"{epoch+1} epochs."
        )

        break


# ============================================================
# 17. TRAINING COMPLETE
# ============================================================

total_time = time.time() - start_time

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy*100:.2f}%"
)

print(
    f"Best Validation Macro F1 : "
    f"{best_val_f1*100:.2f}%"
)

print(
    f"Best Epoch               : "
    f"{best_epoch}"
)

print(
    f"Training Time            : "
    f"{total_time/60:.2f} minutes"
)

print()
print("Best model saved to:")
print(MODEL_PATH)
print()


# ============================================================
# 18. RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )

else:

    print(
        "WARNING: No improvement was detected."
    )


# ============================================================
# 19. FINAL VALIDATION REPORT
# ============================================================

print("=" * 70)
print("FINAL VALIDATION REPORT")
print("=" * 70)

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


final_val_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

final_val_macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro"
)

print(
    f"Validation Accuracy : "
    f"{final_val_accuracy*100:.2f}%"
)

print(
    f"Validation Macro F1 : "
    f"{final_val_macro_f1*100:.2f}%"
)

print()

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )
)

print("Validation Confusion Matrix:")
print()

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)

print()
print("=" * 70)
print("DONE")
print("=" * 70)