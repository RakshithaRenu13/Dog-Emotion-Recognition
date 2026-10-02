from pathlib import Path
import random
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
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAINVAL_DIR = Path("Dataset_segmented_trainval")
TEST_DIR = Path("Dataset_segmented_test")

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "dinov2_cls_mean_patch_svm_tuned.joblib"

BATCH_SIZE = 16
VAL_SPLIT = 0.20

CLASSES = ["angry", "curious", "happy", "sad", "sleepy"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("DINOv2 CLS + MEAN PATCH SVM HYPERPARAMETER TUNING")
print("=" * 70)

print(f"Using device: {DEVICE}")


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
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

trainval_dataset = datasets.ImageFolder(
    root=str(TRAINVAL_DIR),
    transform=transform
)

test_dataset = datasets.ImageFolder(
    root=str(TEST_DIR),
    transform=transform
)

print(f"Train/Validation images: {len(trainval_dataset)}")
print(f"Test images: {len(test_dataset)}")

print(f"Classes: {trainval_dataset.classes}")


# ============================================================
# CREATE SAME 80/20 SPLIT
# ============================================================

indices = np.arange(len(trainval_dataset))
labels = np.array(trainval_dataset.targets)

train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=labels
)

train_subset = Subset(trainval_dataset, train_idx)
val_subset = Subset(trainval_dataset, val_idx)

print(f"\nTraining images: {len(train_subset)}")
print(f"Validation images: {len(val_subset)}")


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_subset,
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

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=True
)

dinov2 = dinov2.to(DEVICE)
dinov2.eval()

print("DINOv2 loaded successfully.")


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

@torch.no_grad()
def extract_features(loader, name):

    all_features = []
    all_labels = []

    print(f"\nExtracting {name} features...")

    total = len(loader)

    for batch_num, (images, labels_batch) in enumerate(loader):

        images = images.to(DEVICE)

        output = dinov2.forward_features(images)

        # CLS token
        cls_features = output["x_norm_clstoken"]

        # Patch tokens
        patch_tokens = output["x_norm_patchtokens"]

        # Mean pooling over all patch tokens
        mean_patch_features = patch_tokens.mean(dim=1)

        # Concatenate CLS + Mean Patch
        combined_features = torch.cat(
            [cls_features, mean_patch_features],
            dim=1
        )

        all_features.append(
            combined_features.cpu().numpy()
        )

        all_labels.append(
            labels_batch.numpy()
        )

        if (batch_num + 1) % 20 == 0 or (batch_num + 1) == total:
            print(
                f"  Processed {batch_num + 1}/{total} batches",
                end="\r"
            )

    print()

    features = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)

    print(f"{name} feature shape: {features.shape}")

    return features, labels_array


# ============================================================
# EXTRACT FEATURES
# ============================================================

X_train, y_train = extract_features(
    train_loader,
    "TRAIN"
)

X_val, y_val = extract_features(
    val_loader,
    "VALIDATION"
)

X_test, y_test = extract_features(
    test_loader,
    "TEST"
)


# ============================================================
# SVM HYPERPARAMETERS
# ============================================================

C_VALUES = [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0
]

GAMMA_VALUES = [
    "scale",
    0.001,
    0.01,
    0.1
]


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

print("\n")
print("=" * 70)
print("SVM HYPERPARAMETER SEARCH")
print("=" * 70)

best_accuracy = 0.0
best_macro_f1 = 0.0
best_C = None
best_gamma = None

results = []

total_experiments = len(C_VALUES) * len(GAMMA_VALUES)
experiment_number = 0


for C in C_VALUES:

    for gamma in GAMMA_VALUES:

        experiment_number += 1

        print(
            f"\n[{experiment_number}/{total_experiments}] "
            f"C={C}, gamma={gamma}"
        )

        svm_model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                class_weight="balanced"
            )
        )

        print("Training SVM...", end=" ")

        svm_model.fit(X_train, y_train)

        predictions = svm_model.predict(X_val)

        accuracy = accuracy_score(
            y_val,
            predictions
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro"
        )

        print(
            f"Accuracy={accuracy * 100:.2f}% | "
            f"Macro F1={macro_f1 * 100:.2f}%"
        )

        results.append({
            "C": C,
            "gamma": gamma,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

        # Select based primarily on Macro F1
        if macro_f1 > best_macro_f1:

            best_macro_f1 = macro_f1
            best_accuracy = accuracy
            best_C = C
            best_gamma = gamma


# ============================================================
# PRINT SEARCH RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("HYPERPARAMETER SEARCH RESULTS")
print("=" * 70)

sorted_results = sorted(
    results,
    key=lambda x: x["macro_f1"],
    reverse=True
)

for i, result in enumerate(sorted_results):

    print(
        f"{i + 1:2d}. "
        f"C={result['C']:<5} "
        f"gamma={str(result['gamma']):<8} "
        f"Accuracy={result['accuracy'] * 100:.2f}% "
        f"Macro F1={result['macro_f1'] * 100:.2f}%"
    )


# ============================================================
# BEST VALIDATION MODEL
# ============================================================

print("\n")
print("=" * 70)
print("BEST VALIDATION CONFIGURATION")
print("=" * 70)

print(f"Best C:       {best_C}")
print(f"Best gamma:   {best_gamma}")
print(f"Val Accuracy: {best_accuracy * 100:.2f}%")
print(f"Val Macro F1: {best_macro_f1 * 100:.2f}%")


# ============================================================
# RETRAIN BEST MODEL ON TRAIN + VALIDATION
# ============================================================

print("\n")
print("=" * 70)
print("RETRAINING BEST MODEL ON FULL TRAIN + VALIDATION DATA")
print("=" * 70)

X_train_full = np.concatenate(
    [X_train, X_val],
    axis=0
)

y_train_full = np.concatenate(
    [y_train, y_val],
    axis=0
)

print(f"Full training feature shape: {X_train_full.shape}")

final_model = make_pipeline(
    StandardScaler(),
    SVC(
        kernel="rbf",
        C=best_C,
        gamma=best_gamma,
        class_weight="balanced"
    )
)

print("Training final SVM...")

final_model.fit(
    X_train_full,
    y_train_full
)

print("Final SVM training complete.")


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_predictions = final_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

print("\n")
print("=" * 70)
print(
    f"TUNED DINOv2 CLS + MEAN PATCH + SVM TEST ACCURACY: "
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
        target_names=CLASSES,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

artifact = {
    "model": final_model,
    "classes": CLASSES,
    "feature_type": "DINOv2 CLS token + Mean Patch tokens",
    "feature_dimension": 768,
    "svm_kernel": "rbf",
    "C": best_C,
    "gamma": best_gamma,
    "validation_accuracy": best_accuracy,
    "validation_macro_f1": best_macro_f1,
    "test_accuracy": test_accuracy,
    "test_macro_f1": test_macro_f1,
    "seed": SEED
}

joblib.dump(
    artifact,
    MODEL_PATH
)

print("\n")
print("=" * 70)
print("FINAL MODEL SAVED")
print("=" * 70)

print(f"Path: {MODEL_PATH}")
print(f"Best C: {best_C}")
print(f"Best gamma: {best_gamma}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Macro F1: {test_macro_f1 * 100:.2f}%")

print("=" * 70)