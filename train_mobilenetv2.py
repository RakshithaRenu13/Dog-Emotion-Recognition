import os
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


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

TEST_DIR = os.path.join(
    BASE_DIR,
    "Dataset_test_clean"
    
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "mobilenetv2_final.pth"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

PATIENCE = 7

NUM_WORKERS = 0
PIN_MEMORY = True


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("MOBILENETV2 DOG EMOTION RECOGNITION")
print("=" * 70)

print(f"Device: {device}")

if torch.cuda.is_available():
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

print()


# ============================================================
# 3. CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR}"
    )

if not os.path.exists(TEST_DIR):
    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ============================================================
# 4. TRANSFORMS
# ============================================================

# Same general preprocessing policy as the final
# EfficientNet experiment.
#
# Training:
# moderate augmentation
#
# Validation/Test:
# deterministic 224 x 224 preprocessing

train_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

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


val_test_transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

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
    transform=val_test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=val_test_transform
)

classes = train_dataset.classes
num_classes = len(classes)


print("DATASET INFORMATION")
print("-" * 70)

print(
    f"Train images      : {len(train_dataset)}"
)

print(
    f"Validation images : {len(val_dataset)}"
)

print(
    f"Test images       : {len(test_dataset)}"
)

print(
    f"Classes           : {classes}"
)

print(
    f"Number of classes : {num_classes}"
)

print()


# ============================================================
# 6. VERIFY CLASS ORDER
# ============================================================

if train_dataset.classes != val_dataset.classes:

    raise ValueError(
        "Training and validation class order does not match!\n"
        f"Train: {train_dataset.classes}\n"
        f"Val:   {val_dataset.classes}"
    )


if train_dataset.classes != test_dataset.classes:

    raise ValueError(
        "Training and test class order does not match!\n"
        f"Train: {train_dataset.classes}\n"
        f"Test:  {test_dataset.classes}"
    )


# ============================================================
# 7. TRAINING CLASS DISTRIBUTION
# ============================================================

train_targets = np.array(
    train_dataset.targets
)

class_counts = np.bincount(
    train_targets,
    minlength=num_classes
)


print("TRAINING CLASS DISTRIBUTION")
print("-" * 70)

for class_name, count in zip(
    classes,
    class_counts
):

    print(
        f"{class_name:10s}: {count}"
    )

print()


# ============================================================
# 8. CLASS WEIGHTS
# ============================================================

class_weights = (
    len(train_dataset)
    /
    (
        num_classes
        *
        class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(device)


print("CLASS WEIGHTS")
print("-" * 70)

for class_name, weight in zip(
    classes,
    class_weights.cpu().numpy()
):

    print(
        f"{class_name:10s}: {weight:.4f}"
    )

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)


# ============================================================
# 10. LOAD PRETRAINED MOBILENETV2
# ============================================================

print("Loading pretrained MobileNetV2...")

weights = models.MobileNet_V2_Weights.DEFAULT

model = models.mobilenet_v2(
    weights=weights
)


# ============================================================
# 11. REPLACE CLASSIFIER
# ============================================================

in_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(device)


print("MobileNetV2 loaded.")
print(
    f"Classifier input features: {in_features}"
)
print()


# ============================================================
# 12. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=LABEL_SMOOTHING
)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# 14. LEARNING RATE SCHEDULER
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ============================================================
# 15. TRAIN FUNCTION
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

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            *
            images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = (
        running_loss / total
    )

    epoch_accuracy = (
        correct / total
    )

    return (
        epoch_loss,
        epoch_accuracy
    )


# ============================================================
# 16. VALIDATION FUNCTION
# ============================================================

def evaluate(
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
                loss.item()
                *
                images.size(0)
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

    loss = (
        running_loss / total
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro"
    )

    return (
        loss,
        accuracy,
        macro_f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 17. TRAINING
# ============================================================

print("=" * 70)
print("STARTING MOBILENETV2 TRAINING")
print("=" * 70)

print(
    f"Learning rate   : {LEARNING_RATE}"
)

print(
    f"Weight decay    : {WEIGHT_DECAY}"
)

print(
    f"Label smoothing : {LABEL_SMOOTHING}"
)

print(
    f"Batch size      : {BATCH_SIZE}"
)

print(
    f"Maximum epochs  : {NUM_EPOCHS}"
)

print(
    f"Early stopping  : {PATIENCE}"
)

print()


best_val_f1 = 0.0
best_val_accuracy = 0.0
best_epoch = 0

best_model_state = None

epochs_without_improvement = 0

start_time = time.time()


# ============================================================
# 18. EPOCH LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()

    # ------------------------------
    # Training
    # ------------------------------

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    # ------------------------------
    # Validation
    # ------------------------------

    (
        val_loss,
        val_accuracy,
        val_macro_f1,
        _,
        _
    ) = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    # ------------------------------
    # Scheduler
    # ------------------------------

    scheduler.step(
        val_macro_f1
    )

    current_lr = optimizer.param_groups[0]["lr"]

    epoch_time = (
        time.time()
        -
        epoch_start
    )

    # ------------------------------
    # Best model
    # ------------------------------

    if val_macro_f1 > best_val_f1:

        best_val_f1 = val_macro_f1

        best_val_accuracy = (
            val_accuracy
        )

        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        # Save checkpoint
        torch.save(
            {
                "model_state_dict":
                    best_model_state,

                "classes":
                    classes,

                "num_classes":
                    num_classes,

                "image_size":
                    IMAGE_SIZE,

                "best_val_accuracy":
                    best_val_accuracy,

                "best_val_macro_f1":
                    best_val_f1,

                "best_epoch":
                    best_epoch
            },
            MODEL_PATH
        )

        epochs_without_improvement = 0

        marker = "<-- BEST"

    else:

        epochs_without_improvement += 1

        marker = ""

    # ------------------------------
    # Print
    # ------------------------------

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

    # ------------------------------
    # Early stopping
    # ------------------------------

    if (
        epochs_without_improvement
        >=
        PATIENCE
    ):

        print()

        print(
            f"Early stopping triggered "
            f"after {epoch+1} epochs."
        )

        break


# ============================================================
# 19. TRAINING COMPLETE
# ============================================================

total_time = (
    time.time()
    -
    start_time
)

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

print(
    "Best model saved to:"
)

print(MODEL_PATH)

print()


# ============================================================
# 20. RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )


# ============================================================
# 21. FINAL VALIDATION REPORT
# ============================================================

print("=" * 70)
print("FINAL VALIDATION REPORT")
print("=" * 70)

(
    final_val_loss,
    final_val_accuracy,
    final_val_macro_f1,
    val_labels,
    val_predictions
) = evaluate(
    model,
    val_loader,
    criterion,
    device
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
        val_labels,
        val_predictions,
        target_names=classes,
        digits=4
    )
)

print(
    "Validation Confusion Matrix:"
)

print()

print(
    confusion_matrix(
        val_labels,
        val_predictions
    )
)

print()


# ============================================================
# 22. FINAL TEST EVALUATION
# ============================================================

print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

(
    test_loss,
    test_accuracy,
    test_macro_f1,
    test_labels,
    test_predictions
) = evaluate(
    model,
    test_loader,
    criterion,
    device
)


print()

print(
    f"Test Accuracy : "
    f"{test_accuracy*100:.2f}%"
)

print(
    f"Test Macro F1 : "
    f"{test_macro_f1*100:.2f}%"
)

print()

print("Classification Report:")
print()

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=classes,
        digits=4
    )
)

print()

print("Test Confusion Matrix:")
print()

print(
    confusion_matrix(
        test_labels,
        test_predictions
    )
)

print()


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("=" * 70)
print("MOBILENETV2 FINAL SUMMARY")
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
    f"Test Accuracy            : "
    f"{test_accuracy*100:.2f}%"
)

print(
    f"Test Macro F1            : "
    f"{test_macro_f1*100:.2f}%"
)

print()

print("Model:")
print(MODEL_PATH)

print()

print("=" * 70)
print("DONE")
print("=" * 70)
