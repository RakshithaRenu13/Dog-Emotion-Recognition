from pathlib import Path
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

TRAINVAL_DIR = Path("Dataset_segmented_trainval")
TEST_DIR = Path("Dataset_segmented_test")

BATCH_SIZE = 16
NUM_EPOCHS = 20

NUM_CLASSES = 5

CLASSES = [
    "angry",
    "curious",
    "happy",
    "sad",
    "sleepy"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DINOv2 FINE-TUNING")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.80, 1.0),
        ratio=(0.90, 1.10)
    ),
    transforms.RandomHorizontalFlip(
        p=0.5
    ),
    transforms.RandomRotation(
        degrees=12
    ),

    transforms.ColorJitter(
        brightness=0.200,
        contrast=0.20,
        saturation=0.15,
        hue=0.03
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# DATASETS
# ============================================================

print("\nLoading datasets...")

# Same directory, different transforms
train_dataset_full = datasets.ImageFolder(
    root=str(TRAINVAL_DIR),
    transform=train_transform
)

val_dataset_full = datasets.ImageFolder(
    root=str(TRAINVAL_DIR),
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=val_transform
)

print("Total train/val:", len(train_dataset_full))
print("Test:", len(test_dataset))


# ============================================================
# SAME SPLIT AS PREVIOUS EXPERIMENTS
# ============================================================

from sklearn.model_selection import train_test_split

indices = np.arange(
    len(train_dataset_full)
)

labels = np.array(
    train_dataset_full.targets
)

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)


# ============================================================
# SUBSET
# ============================================================

train_dataset = torch.utils.data.Subset(
    train_dataset_full,
    train_idx
)

val_dataset = torch.utils.data.Subset(
    val_dataset_full,
    val_idx
)


print("Training:", len(train_dataset))
print("Validation:", len(val_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD DINOv2
# ============================================================

print("\nLoading DINOv2 ViT-S/14...")

backbone = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=True
)

backbone = backbone.to(DEVICE)

print("DINOv2 loaded.")


# ============================================================
# FREEZE ENTIRE BACKBONE
# ============================================================

for param in backbone.parameters():
    param.requires_grad = False


# ============================================================
# UNFREEZE LAST 4 TRANSFORMER BLOCKS
# ============================================================

print("\nUnfreezing last 8 DINOv2 blocks...")

for block in backbone.blocks[-8:]:
    for param in block.parameters():
        param.requires_grad = True


# Also allow final normalization to adapt
for param in backbone.norm.parameters():
    param.requires_grad = True


# ============================================================
# MULTIMODAL-STYLE FEATURE CLASSIFIER
# CLS + MEAN PATCH
# ============================================================

class DINOEmotionModel(nn.Module):

    def __init__(self, backbone, num_classes):

        super().__init__()

        self.backbone = backbone

        self.classifier = nn.Sequential(

            nn.Linear(768, 256),

            nn.LayerNorm(256),

            nn.GELU(),

            nn.Dropout(0.35),

            nn.Linear(256, 128),

            nn.GELU(),

            nn.Dropout(0.25),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        output = self.backbone.forward_features(x)

        # CLS token
        cls_features = output[
            "x_norm_clstoken"
        ]

        # Patch tokens
        patch_features = output[
            "x_norm_patchtokens"
        ]

        # Mean patch representation
        mean_patch = patch_features.mean(
            dim=1
        )

        # CLS + Mean Patch
        features = torch.cat(
            [
                cls_features,
                mean_patch
            ],
            dim=1
        )

        return self.classifier(features)


# ============================================================
# CREATE MODEL
# ============================================================

model = DINOEmotionModel(
    backbone,
    NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# COUNT PARAMETERS
# ============================================================

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total_params = sum(
    p.numel()
    for p in model.parameters()
)

print("\nTotal parameters:",
      f"{total_params:,}")

print(
    "Trainable parameters:",
    f"{trainable_params:,}"
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = labels[train_idx]

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("\nClass counts:")
for i, cls in enumerate(CLASSES):
    print(
        f"{cls}: {class_counts[i]}"
    )


# Balanced class weights
class_weights = (
    len(train_labels)
    /
    (
        NUM_CLASSES *
        class_counts
    )
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")
print(class_weights)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)


# ============================================================
# OPTIMIZER
# ============================================================

# Very small LR for pretrained DINOv2
backbone_params = [
    p for p in backbone.parameters()
    if p.requires_grad
]

classifier_params = [
    p for p in model.classifier.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    [
        {
            "params": backbone_params,
            "lr": 7e-6
        },
        {
            "params": classifier_params,
            "lr": 5e-4
        }
    ],
    weight_decay=0.05
)


# ============================================================
# SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS
)


# ============================================================
# MIXED PRECISION
# ============================================================

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available()
)


# ============================================================
# TRAINING
# ============================================================

best_val_f1 = 0.0
best_val_acc = 0.0
best_state = None

print("\n")
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)


for epoch in range(NUM_EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_losses = []
    train_preds = []
    train_targets = []

    for images, targets in train_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        targets = targets.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

            loss = criterion(
                outputs,
                targets
            )

        scaler.scale(loss).backward()

        # Prevent exploding gradients
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        train_losses.append(
            loss.item()
        )

        predictions = outputs.argmax(
            dim=1
        )

        train_preds.extend(
            predictions.detach().cpu().numpy()
        )

        train_targets.extend(
            targets.detach().cpu().numpy()
        )


    # --------------------------------------------------------
    # TRAIN METRICS
    # --------------------------------------------------------

    train_acc = accuracy_score(
        train_targets,
        train_preds
    )

    train_f1 = f1_score(
        train_targets,
        train_preds,
        average="macro"
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_preds = []
    val_targets = []
    val_losses = []

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                loss = criterion(
                    outputs,
                    targets
                )

            val_losses.append(
                loss.item()
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_preds.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                targets.cpu().numpy()
            )


    val_acc = accuracy_score(
        val_targets,
        val_preds
    )

    val_f1 = f1_score(
        val_targets,
        val_preds,
        average="macro"
    )


    scheduler.step()


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print(
        f"Train Loss: {np.mean(train_losses):.4f}"
    )

    print(
        f"Train Accuracy: {train_acc * 100:.2f}%"
    )

    print(
        f"Train Macro F1: {train_f1 * 100:.2f}%"
    )

    print(
        f"Val Loss: {np.mean(val_losses):.4f}"
    )

    print(
        f"Val Accuracy: {val_acc * 100:.2f}%"
    )

    print(
        f"Val Macro F1: {val_f1 * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_val_acc = val_acc

        best_state = {
            k: v.cpu().clone()
            for k, v in model.state_dict().items()
        }

        print(
            ">>> BEST MODEL UPDATED"
        )


# ============================================================
# RESTORE BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("BEST VALIDATION RESULT")
print("=" * 70)

print(
    f"Validation Accuracy: "
    f"{best_val_acc * 100:.2f}%"
)

print(
    f"Validation Macro F1: "
    f"{best_val_f1 * 100:.2f}%"
)

model.load_state_dict(
    best_state
)

model = model.to(DEVICE)


# ============================================================
# TEST
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

model.eval()

test_preds = []
test_targets = []

with torch.no_grad():

    for images, targets in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        with torch.amp.autocast(
            device_type="cuda",
            enabled=torch.cuda.is_available()
        ):

            outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        test_preds.extend(
            predictions.cpu().numpy()
        )

        test_targets.extend(
            targets.numpy()
        )


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_targets,
    test_preds
)

test_macro_f1 = f1_score(
    test_targets,
    test_preds,
    average="macro"
)


print("\n")
print("=" * 70)

print(
    f"TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"TEST MACRO F1: "
    f"{test_macro_f1 * 100:.2f}%"
)

print("=" * 70)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        test_targets,
        test_preds,
        target_names=CLASSES,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    test_targets,
    test_preds
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_PATH = (
    Path("models")
    / "dinov2_finetuned_emotion.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "classes":
            CLASSES,

        "test_accuracy":
            test_accuracy,

        "test_macro_f1":
            test_macro_f1,

        "val_accuracy":
            best_val_acc,

        "val_macro_f1":
            best_val_f1,

        "epochs":
            NUM_EPOCHS
    },
    MODEL_PATH
)


print("\n")
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"Path: {MODEL_PATH}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{test_macro_f1 * 100:.2f}%"
)

print("=" * 70)