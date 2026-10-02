# ============================================================
# DINOv2 + CLS + MEAN PATCH SVM
# FINAL TEST EVALUATION
# ============================================================

import random
from pathlib import Path

import joblib
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIG
# ============================================================

SEED = 42

TEST_DIR = Path("Dataset_segmented_test")

TRAINVAL_DIR = Path("Dataset_segmented_trainval")

BATCH_SIZE = 16

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
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# LOAD TRAIN/VALIDATION DATASET
# ============================================================

trainval_dataset = datasets.ImageFolder(
    root=str(TRAINVAL_DIR),
    transform=transform
)

print(
    "Train/Validation images:",
    len(trainval_dataset)
)

if trainval_dataset.classes != CLASSES:
    raise ValueError(
        f"Expected {CLASSES}, "
        f"got {trainval_dataset.classes}"
    )


# ============================================================
# RECREATE TRAIN/VALIDATION SPLIT
# ============================================================

from sklearn.model_selection import train_test_split

labels = np.array(
    trainval_dataset.targets
)

indices = np.arange(
    len(trainval_dataset)
)

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=labels
)


# ============================================================
# LOAD TEST DATASET
# ============================================================

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_DIR}"
    )

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=transform
)

print(
    "Test images:",
    len(test_dataset)
)

if test_dataset.classes != CLASSES:
    raise ValueError(
        f"Expected {CLASSES}, "
        f"got {test_dataset.classes}"
    )


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

def extract_features(dataset, indices=None):

    if indices is not None:

        from torch.utils.data import Subset

        dataset_used = Subset(
            dataset,
            indices.tolist()
        )

    else:

        dataset_used = dataset


    loader = DataLoader(
        dataset_used,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )


    features = []

    labels = []


    with torch.inference_mode():

        for batch_idx, (
            images,
            batch_labels
        ) in enumerate(loader):

            images = images.to(device)

            output = dinov2.forward_features(
                images
            )

            # --------------------------------------------
            # CLS TOKEN
            # --------------------------------------------

            cls_token = output[
                "x_norm_clstoken"
            ]

            # --------------------------------------------
            # PATCH TOKENS
            # --------------------------------------------

            patch_tokens = output[
                "x_norm_patchtokens"
            ]

            # --------------------------------------------
            # MEAN PATCH
            # --------------------------------------------

            mean_patch = patch_tokens.mean(
                dim=1
            )

            # --------------------------------------------
            # CONCATENATE
            # --------------------------------------------

            combined = torch.cat(
                [
                    cls_token,
                    mean_patch
                ],
                dim=1
            )

            features.append(
                combined.cpu().numpy()
            )

            labels.extend(
                batch_labels.numpy()
            )

            if (batch_idx + 1) % 20 == 0:

                print(
                    f"Processed "
                    f"{batch_idx + 1}/"
                    f"{len(loader)} batches"
                )


    return (
        np.concatenate(features),
        np.asarray(labels)
    )


# ============================================================
# EXTRACT TRAINING FEATURES
# ============================================================

print("\nExtracting training features...")

X_train, y_train = extract_features(
    trainval_dataset,
    train_idx
)

print(
    "Training feature shape:",
    X_train.shape
)


# ============================================================
# EXTRACT VALIDATION FEATURES
# ============================================================

print("\nExtracting validation features...")

X_val, y_val = extract_features(
    trainval_dataset,
    val_idx
)

print(
    "Validation feature shape:",
    X_val.shape
)


# ============================================================
# EXTRACT TEST FEATURES
# ============================================================

print("\nExtracting TEST features...")

X_test, y_test = extract_features(
    test_dataset
)

print(
    "Test feature shape:",
    X_test.shape
)


# ============================================================
# TRAIN FINAL SVM
# ============================================================

print("\nTraining CLS + Mean Patch SVM...")

svm_model = make_pipeline(
    StandardScaler(),

    SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced"
    )
)

svm_model.fit(
    X_train,
    y_train
)


# ============================================================
# VALIDATION CHECK
# ============================================================

val_predictions = svm_model.predict(
    X_val
)

val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_macro_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

print("\n" + "=" * 60)

print(
    f"Validation Accuracy: "
    f"{val_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro F1: "
    f"{val_macro_f1 * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# TEST PREDICTION
# ============================================================

print("\nEvaluating on TEST data...")

test_predictions = svm_model.predict(
    X_test
)


# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)


print("\n" + "=" * 70)

print(
    f"CLS + MEAN PATCH DINOv2 + SVM TEST ACCURACY: "
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
        y_test,
        test_predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=list(range(len(CLASSES)))
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")

plt.title(
    "DINOv2 CLS + Mean Patch + SVM"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix_dinov2_cls_mean_svm.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE MODEL
# ============================================================

artifact = {

    "model_name":
        "dinov2_vits14_cls_mean_patch_svm",

    "class_names":
        CLASSES,

    "svm_pipeline":
        svm_model,

    "feature_dimension":
        768,

    "validation_accuracy":
        float(val_accuracy),

    "validation_macro_f1":
        float(val_macro_f1),

    "test_accuracy":
        float(test_accuracy),

    "test_macro_f1":
        float(test_macro_f1),

    "seed":
        SEED,

    "image_size":
        224,

    "dinov2_name":
        "dinov2_vits14",

    "feature_type":
        "CLS + Mean Patch"

}


MODEL_PATH = Path(
    "models/dinov2_cls_mean_patch_svm.joblib"
)

joblib.dump(
    artifact,
    MODEL_PATH
)


print("\n" + "=" * 70)

print("FINAL MODEL SAVED")

print(
    "Path:",
    MODEL_PATH
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