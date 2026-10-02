
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42
DATA_DIR = Path("Dataset_segmented_trainval")
OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "dinov2_svm_classifier.joblib"

BATCH_SIZE = 16
VAL_SPLIT = 0.20

CLASSES = ["angry", "curious", "happy", "sad", "sleepy"]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_DIR}")

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
        f"Expected classes {CLASSES}, got {dataset.classes}"
    )

labels = np.array(dataset.targets)

# Stratified split: preserve class proportions.
indices = np.arange(len(dataset))
train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=labels
)

print("Training images:", len(train_idx))
print("Validation images:", len(val_idx))

# --------------------------------------------------
# LOAD PRETRAINED DINOV2
# --------------------------------------------------

print("\nLoading pretrained DINOv2 ViT-S/14...")
print("The first run may download model weights.")

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=True
)

dinov2 = dinov2.to(device)
dinov2.eval()

# --------------------------------------------------
# FEATURE EXTRACTION
# --------------------------------------------------

def extract_features(indices, description):
    subset = Subset(dataset, indices.tolist())

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    features = []
    extracted_labels = []

    with torch.inference_mode():
        for batch_idx, (images, batch_labels) in enumerate(loader):
            images = images.to(device)

            output = dinov2.forward_features(images)
            batch_features = output["x_norm_clstoken"]

            features.append(
                batch_features.cpu().numpy()
            )
            extracted_labels.extend(batch_labels.numpy())

            if (batch_idx + 1) % 20 == 0:
                print(
                    f"{description}: "
                    f"{batch_idx + 1}/{len(loader)} batches"
                )

    return (
        np.concatenate(features, axis=0),
        np.asarray(extracted_labels)
    )

print("\nExtracting training features...")
X_train, y_train = extract_features(train_idx, "Train")

print("\nExtracting validation features...")
X_val, y_val = extract_features(val_idx, "Validation")

print("Feature shape:", X_train.shape)

# --------------------------------------------------
# TRAIN SVM
# --------------------------------------------------

print("\nTraining SVM classifier...")

# svm_model = make_pipeline(
#     StandardScaler(),
#     SVC(
#         kernel="rbf",
#         C=1.0,
#         gamma="scale",
#         class_weight="balanced"
#     )
# )
# ============================================================
# DINOv2 + SVM HYPERPARAMETER SEARCH
# ============================================================

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

print("\n" + "=" * 70)
print("DINOv2 + SVM HYPERPARAMETER SEARCH")
print("=" * 70)

# ------------------------------------------------------------
# Different SVM configurations
# ------------------------------------------------------------

configs = [

    # RBF
    {
        "name": "RBF C=0.1",
        "kernel": "rbf",
        "C": 0.1,
        "gamma": "scale",
        "class_weight": "balanced"
    },

    {
        "name": "RBF C=1",
        "kernel": "rbf",
        "C": 1,
        "gamma": "scale",
        "class_weight": "balanced"
    },

    {
        "name": "RBF C=10",
        "kernel": "rbf",
        "C": 10,
        "gamma": "scale",
        "class_weight": "balanced"
    },

    {
        "name": "RBF C=100",
        "kernel": "rbf",
        "C": 100,
        "gamma": "scale",
        "class_weight": "balanced"
    },

    # Linear
    {
        "name": "Linear C=0.1",
        "kernel": "linear",
        "C": 0.1,
        "class_weight": "balanced"
    },

    {
        "name": "Linear C=1",
        "kernel": "linear",
        "C": 1,
        "class_weight": "balanced"
    },

    {
        "name": "Linear C=10",
        "kernel": "linear",
        "C": 10,
        "class_weight": "balanced"
    },

    # RBF without class balancing
    {
        "name": "RBF C=1 no balance",
        "kernel": "rbf",
        "C": 1,
        "gamma": "scale",
        "class_weight": None
    },

    {
        "name": "RBF C=10 no balance",
        "kernel": "rbf",
        "C": 10,
        "gamma": "scale",
        "class_weight": None
    }
]

results = []

# ------------------------------------------------------------
# Train and evaluate each configuration
# ------------------------------------------------------------

for config in configs:

    print("\n" + "-" * 60)
    print("Testing:", config["name"])

    svm = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=config["kernel"],
            C=config["C"],
            gamma=config.get("gamma", "scale"),
            class_weight=config["class_weight"]
        )
    )

    svm.fit(X_train, y_train)

    pred = svm.predict(X_val)

    acc = accuracy_score(y_val, pred)

    macro_f1 = f1_score(
        y_val,
        pred,
        average="macro"
    )

    results.append(
        (
            config["name"],
            acc,
            macro_f1,
            svm
        )
    )

    print(
        f"Accuracy : {acc * 100:.2f}%"
    )

    print(
        f"Macro F1 : {macro_f1 * 100:.2f}%"
    )

# ------------------------------------------------------------
# SORT BY MACRO F1
# ------------------------------------------------------------

results.sort(
    key=lambda x: x[2],
    reverse=True
)

print("\n" + "=" * 70)
print("SVM CONFIGURATION RESULTS")
print("=" * 70)

for name, acc, macro_f1, _ in results:

    print(
        f"{name:25s} | "
        f"Accuracy: {acc * 100:.2f}% | "
        f"Macro F1: {macro_f1 * 100:.2f}%"
    )

# ------------------------------------------------------------
# BEST MODEL
# ------------------------------------------------------------

best_name, best_acc, best_f1, best_svm = results[0]

print("\n" + "=" * 70)
print("BEST SVM")
print("=" * 70)

print("Configuration:", best_name)
print(f"Accuracy: {best_acc * 100:.2f}%")
print(f"Macro F1: {best_f1 * 100:.2f}%")
# svm_model.fit(X_train, y_train)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\nEvaluating on validation data...")

predictions = svm_model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)

print("\n" + "=" * 55)
print(f"DINOv2 + SVM Validation Accuracy: {accuracy * 100:.2f}%")
print("=" * 55)

print("\nClassification Report:")
print(classification_report(
    y_val,
    predictions,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    digits=4,
    zero_division=0
))

cm = confusion_matrix(
    y_val,
    predictions,
    labels=list(range(len(CLASSES)))
)

print("\nConfusion Matrix:")
print(cm)

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

artifact = {
    "model_name": "dinov2_vits14_svm",
    "class_names": CLASSES,
    "svm_pipeline": svm_model,
    "validation_accuracy": float(accuracy),
    "seed": SEED,
    "val_split": VAL_SPLIT,
    "dinov2_name": "dinov2_vits14",
    "image_size": 224,
}

joblib.dump(artifact, MODEL_PATH)

print("\nTraining complete!")
print("Best validation accuracy:", f"{accuracy * 100:.2f}%")
print("Model saved to:", MODEL_PATH)
# ============================================================
# DINOv2 + SVM : COMPLETE VALIDATION ERROR ANALYSIS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

print("\n" + "=" * 70)
print("DINOv2 + SVM COMPLETE ERROR ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# USE YOUR EXISTING VARIABLES
# y_val       = actual validation labels
# predictions  = SVM predictions
# CLASSES      = emotion names
# ------------------------------------------------------------

y_true = y_val
y_pred = predictions
class_names = CLASSES

# ------------------------------------------------------------
# 1. ACCURACY
# ------------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

print(f"\nValidation Accuracy: {accuracy * 100:.2f}%")

# ------------------------------------------------------------
# 2. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ------------------------------------------------------------
# 3. CONFUSION MATRIX
# ------------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names)))
)

print("\nConfusion Matrix:")
print(cm)

# ------------------------------------------------------------
# 4. CONFUSION MATRIX PLOT
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("DINOv2 + SVM Confusion Matrix")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. PER-CLASS PERFORMANCE
# ------------------------------------------------------------

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    zero_division=0
)

print("\n" + "=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

for i, name in enumerate(class_names):

    print(
        f"{name:10s} | "
        f"Precision: {precision[i]:.4f} | "
        f"Recall: {recall[i]:.4f} | "
        f"F1: {f1[i]:.4f} | "
        f"Samples: {support[i]}"
    )

# ------------------------------------------------------------
# 6. MACRO F1
# ------------------------------------------------------------

macro_f1 = np.mean(f1)

print("\nMacro F1:", f"{macro_f1 * 100:.2f}%")

# ------------------------------------------------------------
# 7. MOST CONFUSED EMOTION PAIRS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MOST CONFUSED EMOTION PAIRS")
print("=" * 70)

confusions = []

for actual in range(len(class_names)):

    for predicted in range(len(class_names)):

        if actual != predicted and cm[actual, predicted] > 0:

            confusions.append(
                (
                    cm[actual, predicted],
                    class_names[actual],
                    class_names[predicted]
                )
            )

confusions.sort(reverse=True)

for count, actual, predicted in confusions[:10]:

    print(
        f"{actual:10s} -> "
        f"{predicted:10s} : "
        f"{count} images"
    )

# ------------------------------------------------------------
# 8. BEST AND WORST F1
# ------------------------------------------------------------

best_idx = np.argmax(f1)
worst_idx = np.argmin(f1)

print("\n" + "=" * 70)
print("CLASS PERFORMANCE SUMMARY")
print("=" * 70)

print(
    f"Highest F1 : {class_names[best_idx]} "
    f"({f1[best_idx] * 100:.2f}%)"
)

print(
    f"Lowest F1  : {class_names[worst_idx]} "
    f"({f1[worst_idx] * 100:.2f}%)"
)

# ------------------------------------------------------------
# 9. NORMALIZED CONFUSION MATRIX
# ------------------------------------------------------------

cm_normalized = cm.astype(float) / cm.sum(
    axis=1,
    keepdims=True
)

cm_normalized = np.nan_to_num(cm_normalized)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Normalized Confusion Matrix - DINOv2 + SVM")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Macro F1 : {macro_f1 * 100:.2f}%")

if confusions:

    count, actual, predicted = confusions[0]

    print(
        f"Top confusion : "
        f"{actual} -> {predicted} "
        f"({count} images)"
    )

print("=" * 70)

