import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIG
# ============================================================

SEED = 42

DATA_DIR = Path("Dataset_segmented_trainval")

BATCH_SIZE = 16
VAL_SPLIT = 0.20

CLASSES = [
    "angry",
    "curious",
    "happy",
    "sad",
    "sleepy"
]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

dataset = datasets.ImageFolder(
    root=str(DATA_DIR),
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))

if dataset.classes != CLASSES:
    raise ValueError(
        f"Expected {CLASSES}, got {dataset.classes}"
    )

labels = np.array(dataset.targets)

indices = np.arange(len(dataset))

train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=labels
)

print("Training images:", len(train_idx))
print("Validation images:", len(val_idx))


# ============================================================
# LOAD DINOv2
# ============================================================

print("\nLoading DINOv2 ViT-S/14...")

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=True
)

dinov2 = dinov2.to(device)
dinov2.eval()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(indices, description):

    subset = Subset(
        dataset,
        indices.tolist()
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    cls_features = []
    mean_patch_features = []
    max_patch_features = []

    extracted_labels = []

    with torch.inference_mode():

        for batch_idx, (images, batch_labels) in enumerate(loader):

            images = images.to(device)

            output = dinov2.forward_features(images)

            # ------------------------------------------------
            # CLS TOKEN
            # Shape: [batch, 384]
            # ------------------------------------------------

            cls_token = output["x_norm_clstoken"]

            # ------------------------------------------------
            # PATCH TOKENS
            # Shape: [batch, 256, 384]
            # ------------------------------------------------

            patch_tokens = output["x_norm_patchtokens"]

            # ------------------------------------------------
            # MEAN PATCH FEATURE
            # ------------------------------------------------

            mean_patch = patch_tokens.mean(
                dim=1
            )

            # ------------------------------------------------
            # MAX PATCH FEATURE
            # ------------------------------------------------

            max_patch = patch_tokens.max(
                dim=1
            ).values

            cls_features.append(
                cls_token.cpu().numpy()
            )

            mean_patch_features.append(
                mean_patch.cpu().numpy()
            )

            max_patch_features.append(
                max_patch.cpu().numpy()
            )

            extracted_labels.extend(
                batch_labels.numpy()
            )

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"{description}: "
                    f"{batch_idx + 1}/{len(loader)} batches"
                )

    return (
        np.concatenate(cls_features),
        np.concatenate(mean_patch_features),
        np.concatenate(max_patch_features),
        np.asarray(extracted_labels)
    )


# ============================================================
# EXTRACT FEATURES
# ============================================================

print("\nExtracting training features...")

(
    X_train_cls,
    X_train_mean,
    X_train_max,
    y_train
) = extract_features(
    train_idx,
    "Train"
)


print("\nExtracting validation features...")

(
    X_val_cls,
    X_val_mean,
    X_val_max,
    y_val
) = extract_features(
    val_idx,
    "Validation"
)


# ============================================================
# CREATE FEATURE SETS
# ============================================================

feature_sets = {

    "CLS only":
        (
            X_train_cls,
            X_val_cls
        ),

    "Mean patch only":
        (
            X_train_mean,
            X_val_mean
        ),

    "Max patch only":
        (
            X_train_max,
            X_val_max
        ),

    "CLS + Mean patch":
        (
            np.concatenate(
                [X_train_cls, X_train_mean],
                axis=1
            ),
            np.concatenate(
                [X_val_cls, X_val_mean],
                axis=1
            )
        ),

    "CLS + Max patch":
        (
            np.concatenate(
                [X_train_cls, X_train_max],
                axis=1
            ),
            np.concatenate(
                [X_val_cls, X_val_max],
                axis=1
            )
        ),

    "CLS + Mean + Max":
        (
            np.concatenate(
                [
                    X_train_cls,
                    X_train_mean,
                    X_train_max
                ],
                axis=1
            ),
            np.concatenate(
                [
                    X_val_cls,
                    X_val_mean,
                    X_val_max
                ],
                axis=1
            )
        )
}


# ============================================================
# TRAIN SVM ON EACH FEATURE SET
# ============================================================

results = []

print("\n" + "=" * 75)
print("DINOv2 FEATURE COMPARISON")
print("=" * 75)

for name, (Xtr, Xv) in feature_sets.items():

    print("\n----------------------------------------")
    print(name)
    print("Feature dimension:", Xtr.shape[1])

    svm = make_pipeline(
        StandardScaler(),
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        )
    )

    svm.fit(
        Xtr,
        y_train
    )

    predictions = svm.predict(
        Xv
    )

    accuracy = accuracy_score(
        y_val,
        predictions
    )

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    results.append(
        (
            name,
            Xtr.shape[1],
            accuracy,
            macro_f1
        )
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1 : {macro_f1 * 100:.2f}%"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL FEATURE COMPARISON")
print("=" * 75)

results.sort(
    key=lambda x: x[3],
    reverse=True
)

for name, dimension, accuracy, macro_f1 in results:

    print(
        f"{name:20s} | "
        f"Dim: {dimension:4d} | "
        f"Accuracy: {accuracy * 100:6.2f}% | "
        f"Macro F1: {macro_f1 * 100:6.2f}%"
    )

print("\nBest feature representation:")

best = results[0]

print(
    f"{best[0]} "
    f"({best[1]} dimensions)"
)

print(
    f"Validation Accuracy: "
    f"{best[2] * 100:.2f}%"
)

print(
    f"Validation Macro F1: "
    f"{best[3] * 100:.2f}%"
)
