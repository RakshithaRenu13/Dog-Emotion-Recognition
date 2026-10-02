# ============================================================
# HIERARCHICAL DINOv2 + SVM
#
# Stage 1:
#   5-class classifier
#
# Stage 2:
#   angry / sad / sleepy specialist
#
# Feature:
#   DINOv2 CLS + Mean Patch = 768 dimensions
#
# Model selection:
#   VALIDATION ONLY
#
# Final evaluation:
#   TEST ONCE
# ============================================================

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

MODEL_PATH = (
    OUTPUT_DIR /
    "hierarchical_dinov2_cls_mean_svm.joblib"
)

BATCH_SIZE = 16
VAL_SPLIT = 0.20

CLASSES = [
    "angry",
    "curious",
    "happy",
    "sad",
    "sleepy"
]

# Hard classes
HARD_CLASSES = [
    "angry",
    "sad",
    "sleepy"
]

# Original class indices
# angry=0
# curious=1
# happy=2
# sad=3
# sleepy=4

HARD_CLASS_INDICES = [
    0,
    3,
    4
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PRINT CONFIG
# ============================================================

print("=" * 75)
print("HIERARCHICAL DINOv2 + SVM")
print("=" * 75)

print(f"Device: {DEVICE}")
print(f"Classes: {CLASSES}")
print(f"Hard classes: {HARD_CLASSES}")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

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

print(
    f"Train/Validation images: "
    f"{len(trainval_dataset)}"
)

print(
    f"Test images: "
    f"{len(test_dataset)}"
)

print(
    f"Dataset classes: "
    f"{trainval_dataset.classes}"
)


# ============================================================
# CREATE EXACT SAME TRAIN/VALIDATION SPLIT
# ============================================================

indices = np.arange(
    len(trainval_dataset)
)

labels = np.array(
    trainval_dataset.targets
)

train_idx, val_idx = train_test_split(
    indices,
    test_size=VAL_SPLIT,
    random_state=SEED,
    stratify=labels
)

print("\nTrain/Validation split:")

print(
    f"Training: "
    f"{len(train_idx)}"
)

print(
    f"Validation: "
    f"{len(val_idx)}"
)


# ============================================================
# DATASETS
# ============================================================

train_subset = Subset(
    trainval_dataset,
    train_idx
)

val_subset = Subset(
    trainval_dataset,
    val_idx
)


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
# FEATURE EXTRACTION
# ============================================================

@torch.no_grad()
def extract_features(loader, name):

    features_list = []
    labels_list = []

    print(
        f"\nExtracting {name} features..."
    )

    total_batches = len(loader)

    for batch_number, (images, batch_labels) in enumerate(
        loader,
        start=1
    ):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        output = dinov2.forward_features(
            images
        )

        # ----------------------------------------------------
        # CLS TOKEN
        # ----------------------------------------------------

        cls_features = output[
            "x_norm_clstoken"
        ]

        # ----------------------------------------------------
        # PATCH TOKENS
        # ----------------------------------------------------

        patch_features = output[
            "x_norm_patchtokens"
        ]

        # ----------------------------------------------------
        # MEAN PATCH
        # ----------------------------------------------------

        mean_patch = patch_features.mean(
            dim=1
        )

        # ----------------------------------------------------
        # CONCATENATE
        # ----------------------------------------------------

        combined = torch.cat(
            [
                cls_features,
                mean_patch
            ],
            dim=1
        )

        features_list.append(
            combined.cpu().numpy()
        )

        labels_list.append(
            batch_labels.numpy()
        )

        if (
            batch_number % 20 == 0
            or batch_number == total_batches
        ):

            print(
                f"Processed "
                f"{batch_number}/{total_batches}",
                end="\r"
            )

    print()

    features = np.concatenate(
        features_list,
        axis=0
    )

    labels_array = np.concatenate(
        labels_list,
        axis=0
    )

    print(
        f"{name} features: "
        f"{features.shape}"
    )

    return features, labels_array


# ============================================================
# EXTRACT TRAIN / VAL / TEST
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
# VERIFY FEATURE SIZE
# ============================================================

assert X_train.shape[1] == 768

print(
    "\nFeature dimension verified: 768"
)


# ============================================================
# STAGE 1
#
# ALL FIVE CLASSES
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 1: FIVE-CLASS SVM")
print("=" * 75)


stage1 = make_pipeline(

    StandardScaler(),

    SVC(
        kernel="rbf",

        C=2.0,

        gamma=0.001,

        class_weight="balanced",

        probability=True,

        random_state=SEED
    )
)


print("\nTraining Stage 1...")

stage1.fit(
    X_train,
    y_train
)

print("Stage 1 training complete.")


# ============================================================
# STAGE 1 VALIDATION
# ============================================================

stage1_val_pred = stage1.predict(
    X_val
)

stage1_val_proba = stage1.predict_proba(
    X_val
)

stage1_val_accuracy = accuracy_score(
    y_val,
    stage1_val_pred
)

stage1_val_f1 = f1_score(
    y_val,
    stage1_val_pred,
    average="macro"
)


print("\nStage 1 Validation:")

print(
    f"Accuracy: "
    f"{stage1_val_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{stage1_val_f1 * 100:.2f}%"
)


# ============================================================
# STAGE 2 TRAINING DATA
#
# ONLY angry / sad / sleepy
# ============================================================

print("\n")
print("=" * 75)
print("STAGE 2: HARD-CLASS SPECIALIST")
print("=" * 75)


# ------------------------------------------------------------
# Select hard-class training samples
# ------------------------------------------------------------

hard_train_mask = np.isin(
    y_train,
    HARD_CLASS_INDICES
)

X_hard_train = X_train[
    hard_train_mask
]

y_hard_train_original = y_train[
    hard_train_mask
]


# ------------------------------------------------------------
# Convert labels
#
# angry -> 0
# sad   -> 1
# sleepy-> 2
# ------------------------------------------------------------

hard_label_mapping = {
    0: 0,   # angry
    3: 1,   # sad
    4: 2    # sleepy
}

y_hard_train = np.array([
    hard_label_mapping[x]
    for x in y_hard_train_original
])


print(
    "\nHard training samples:",
    len(X_hard_train)
)

print(
    "Hard feature shape:",
    X_hard_train.shape
)


# ============================================================
# STAGE 2 SVM
# ============================================================

stage2 = make_pipeline(

    StandardScaler(),

    SVC(
        kernel="rbf",

        C=2.0,

        gamma=0.001,

        class_weight="balanced",

        probability=True,

        random_state=SEED
    )
)


print("\nTraining Stage 2...")

stage2.fit(
    X_hard_train,
    y_hard_train
)

print(
    "Stage 2 training complete."
)


# ============================================================
# VALIDATION HARD SAMPLES
# ============================================================

hard_val_mask = np.isin(
    y_val,
    HARD_CLASS_INDICES
)

X_hard_val = X_val[
    hard_val_mask
]

y_hard_val_original = y_val[
    hard_val_mask
]

y_hard_val = np.array([
    hard_label_mapping[x]
    for x in y_hard_val_original
])


stage2_val_pred = stage2.predict(
    X_hard_val
)


stage2_val_accuracy = accuracy_score(
    y_hard_val,
    stage2_val_pred
)

stage2_val_f1 = f1_score(
    y_hard_val,
    stage2_val_pred,
    average="macro"
)


print("\nStage 2 Validation:")

print(
    f"Accuracy: "
    f"{stage2_val_accuracy * 100:.2f}%"
)

print(
    f"Macro F1: "
    f"{stage2_val_f1 * 100:.2f}%"
)


# ============================================================
# HIERARCHICAL PREDICTION FUNCTION
# ============================================================

def hierarchical_predict(
    stage1_model,
    stage2_model,
    X,
    hard_threshold
):

    # --------------------------------------------------------
    # Stage 1 probabilities
    # --------------------------------------------------------

    stage1_proba = (
        stage1_model.predict_proba(X)
    )

    stage1_pred = (
        stage1_proba.argmax(axis=1)
    )

    final_pred = stage1_pred.copy()

    # --------------------------------------------------------
    # Hard-class probability
    #
    # Probability that Stage 1 believes
    # the sample belongs to:
    #
    # angry + sad + sleepy
    # --------------------------------------------------------

    hard_probability = (
        stage1_proba[:, 0]
        +
        stage1_proba[:, 3]
        +
        stage1_proba[:, 4]
    )

    # --------------------------------------------------------
    # Samples requiring Stage 2
    # --------------------------------------------------------

    hard_mask = (
        hard_probability >= hard_threshold
    )

    if np.any(hard_mask):

        hard_features = X[
            hard_mask
        ]

        stage2_predictions = (
            stage2_model.predict(
                hard_features
            )
        )

        # ----------------------------------------------------
        # Convert Stage 2 labels back
        #
        # Stage 2:
        # 0 = angry
        # 1 = sad
        # 2 = sleepy
        # ----------------------------------------------------

        converted_predictions = np.array([
            0 if p == 0 else
            3 if p == 1 else
            4
            for p in stage2_predictions
        ])

        final_pred[
            hard_mask
        ] = converted_predictions

    return final_pred


# ============================================================
# THRESHOLD SEARCH
#
# VALIDATION ONLY
# ============================================================

print("\n")
print("=" * 75)
print("SEARCHING HIERARCHICAL THRESHOLD")
print("=" * 75)


thresholds = np.arange(
    0.30,
    0.96,
    0.02
)


best_threshold = None
best_val_accuracy = 0.0
best_val_f1 = 0.0
best_val_predictions = None


for threshold in thresholds:

    predictions = hierarchical_predict(
        stage1,
        stage2,
        X_val,
        threshold
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

    print(
        f"Threshold={threshold:.2f} | "
        f"Accuracy={accuracy * 100:.2f}% | "
        f"Macro F1={macro_f1 * 100:.2f}%"
    )

    # Primary selection metric = Macro F1
    if macro_f1 > best_val_f1:

        best_val_f1 = macro_f1

        best_val_accuracy = accuracy

        best_threshold = threshold

        best_val_predictions = (
            predictions.copy()
        )


# ============================================================
# BEST HIERARCHICAL VALIDATION RESULT
# ============================================================

print("\n")
print("=" * 75)
print("BEST HIERARCHICAL VALIDATION RESULT")
print("=" * 75)

print(
    f"Best threshold: "
    f"{best_threshold:.2f}"
)

print(
    f"Validation Accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro F1: "
    f"{best_val_f1 * 100:.2f}%"
)


# ============================================================
# VALIDATION CLASSIFICATION REPORT
# ============================================================

print("\nValidation Classification Report:")

print(
    classification_report(
        y_val,
        best_val_predictions,
        target_names=CLASSES,
        digits=4
    )
)


# ============================================================
# RETRAIN STAGE 1 ON TRAIN + VALIDATION
# ============================================================

print("\n")
print("=" * 75)
print("RETRAINING STAGE 1 ON FULL TRAIN + VALIDATION")
print("=" * 75)


X_full = np.concatenate(
    [
        X_train,
        X_val
    ],
    axis=0
)

y_full = np.concatenate(
    [
        y_train,
        y_val
    ],
    axis=0
)


print(
    f"Full training data: "
    f"{X_full.shape}"
)


final_stage1 = make_pipeline(

    StandardScaler(),

    SVC(
        kernel="rbf",

        C=2.0,

        gamma=0.001,

        class_weight="balanced",

        probability=True,

        random_state=SEED
    )
)


print(
    "Training final Stage 1..."
)

final_stage1.fit(
    X_full,
    y_full
)


print(
    "Final Stage 1 complete."
)


# ============================================================
# RETRAIN STAGE 2 ON ALL HARD TRAINING DATA
# ============================================================

print("\n")
print(
    "Retraining Stage 2 on all "
    "angry/sad/sleepy samples..."
)


full_hard_mask = np.isin(
    y_full,
    HARD_CLASS_INDICES
)

X_full_hard = X_full[
    full_hard_mask
]

y_full_hard_original = y_full[
    full_hard_mask
]

y_full_hard = np.array([
    hard_label_mapping[x]
    for x in y_full_hard_original
])


final_stage2 = make_pipeline(

    StandardScaler(),

    SVC(
        kernel="rbf",

        C=2.0,

        gamma=0.001,

        class_weight="balanced",

        probability=True,

        random_state=SEED
    )
)


final_stage2.fit(
    X_full_hard,
    y_full_hard
)


print(
    "Final Stage 2 complete."
)


# ============================================================
# FINAL TEST
#
# THIS IS THE ONLY TEST EVALUATION
# ============================================================

print("\n")
print("=" * 75)
print("FINAL TEST EVALUATION")
print("=" * 75)


test_predictions = hierarchical_predict(
    final_stage1,
    final_stage2,
    X_test,
    best_threshold
)


# ============================================================
# TEST METRICS
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


print("\n")
print("=" * 75)

print(
    f"HIERARCHICAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"HIERARCHICAL TEST MACRO F1: "
    f"{test_macro_f1 * 100:.2f}%"
)

print("=" * 75)


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
# COMPARE WITH CURRENT BEST
# ============================================================

CURRENT_BEST_ACCURACY = 0.7755
CURRENT_BEST_F1 = 0.7669

print("\n")
print("=" * 75)
print("COMPARISON WITH CURRENT BEST")
print("=" * 75)

print(
    f"Current best accuracy: "
    f"{CURRENT_BEST_ACCURACY * 100:.2f}%"
)

print(
    f"Hierarchical accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Accuracy change: "
    f"{(test_accuracy - CURRENT_BEST_ACCURACY) * 100:+.2f}%"
)

print()

print(
    f"Current best Macro F1: "
    f"{CURRENT_BEST_F1 * 100:.2f}%"
)

print(
    f"Hierarchical Macro F1: "
    f"{test_macro_f1 * 100:.2f}%"
)

print(
    f"Macro F1 change: "
    f"{(test_macro_f1 - CURRENT_BEST_F1) * 100:+.2f}%"
)


# ============================================================
# SAVE ARTIFACT
# ============================================================

artifact = {

    "stage1_model":
        final_stage1,

    "stage2_model":
        final_stage2,

    "classes":
        CLASSES,

    "hard_classes":
        HARD_CLASSES,

    "hard_class_indices":
        HARD_CLASS_INDICES,

    "hard_label_mapping":
        hard_label_mapping,

    "feature_type":
        "DINOv2 CLS + Mean Patch",

    "feature_dimension":
        768,

    "svm_kernel":
        "rbf",

    "C":
        2.0,

    "gamma":
        0.001,

    "hard_threshold":
        best_threshold,

    "validation_accuracy":
        best_val_accuracy,

    "validation_macro_f1":
        best_val_f1,

    "test_accuracy":
        test_accuracy,

    "test_macro_f1":
        test_macro_f1,

    "seed":
        SEED
}


joblib.dump(
    artifact,
    MODEL_PATH
)


# ============================================================
# FINAL SAVE MESSAGE
# ============================================================

print("\n")
print("=" * 75)
print("FINAL MODEL SAVED")
print("=" * 75)

print(
    f"Path: {MODEL_PATH}"
)

print(
    f"Threshold: {best_threshold:.2f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{test_macro_f1 * 100:.2f}%"
)

print("=" * 75)