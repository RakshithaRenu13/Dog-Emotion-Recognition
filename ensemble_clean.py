import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

RESNET_PATH = os.path.join(
    MODEL_DIR,
    "resnet18_clean_augmented.pth"
)

EFFICIENTNET_PATH = os.path.join(
    MODEL_DIR,
    "efficientnet_b0_clean_augmented.pth"
)

DINOV2_PATH = os.path.join(
    MODEL_DIR,
    "dinov2_svm_tuned.joblib"
)

RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "clean_ensemble_results.json"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "angry",
    "curious",
    "happy",
    "sad",
    "sleepy"
]

NUM_CLASSES = 5
BATCH_SIZE = 32

# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("CLEAN THREE-MODEL ENSEMBLE")
print("=" * 75)

print("Device:", DEVICE)

# ============================================================
# TRANSFORM
# ============================================================

# Same validation/test preprocessing used for the
# clean ResNet18 and EfficientNet-B0 experiments.

eval_transform = transforms.Compose([
    transforms.Resize((256, 256)),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# DINOv2 also uses 224x224 ImageNet-normalized images.
dino_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DATASETS
# ============================================================

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=eval_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

print("\nValidation images:", len(val_dataset))
print("Clean test images:", len(test_dataset))

print("\nClasses:")
print(val_dataset.classes)

# ============================================================
# LOAD RESNET18
# ============================================================

print("\nLoading ResNet18...")

resnet18 = models.resnet18(
    weights=None
)

resnet18.fc = nn.Linear(
    resnet18.fc.in_features,
    NUM_CLASSES
)

resnet_checkpoint = torch.load(
    RESNET_PATH,
    map_location=DEVICE
)

if isinstance(resnet_checkpoint, dict) and \
   "model_state_dict" in resnet_checkpoint:

    resnet18.load_state_dict(
        resnet_checkpoint["model_state_dict"]
    )

else:

    resnet18.load_state_dict(
        resnet_checkpoint
    )

resnet18 = resnet18.to(DEVICE)
resnet18.eval()

print("ResNet18 loaded.")

# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

print("\nLoading EfficientNet-B0...")

efficientnet = models.efficientnet_b0(
    weights=None
)

efficientnet.classifier[1] = nn.Linear(
    efficientnet.classifier[1].in_features,
    NUM_CLASSES
)

efficientnet_checkpoint = torch.load(
    EFFICIENTNET_PATH,
    map_location=DEVICE
)

if isinstance(efficientnet_checkpoint, dict) and \
   "model_state_dict" in efficientnet_checkpoint:

    efficientnet.load_state_dict(
        efficientnet_checkpoint["model_state_dict"]
    )

else:

    efficientnet.load_state_dict(
        efficientnet_checkpoint
    )

efficientnet = efficientnet.to(DEVICE)
efficientnet.eval()

print("EfficientNet-B0 loaded.")

# ============================================================
# LOAD DINOV2
# ============================================================

print("\nLoading DINOv2...")

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

dinov2 = dinov2.to(DEVICE)
dinov2.eval()

print("DINOv2 loaded.")

# ============================================================
# LOAD DINOv2 SVM PIPELINE
# ============================================================

print("\nLoading DINOv2 SVM pipeline...")

dino_saved = joblib.load(
    DINOV2_PATH
)

dino_pipeline = dino_saved["svm_pipeline"]

print("DINOv2 SVM pipeline loaded.")

# ============================================================
# FUNCTION: CNN PROBABILITIES
# ============================================================

def get_cnn_probabilities(
    model,
    loader,
    model_name
):

    print(
        f"\nGenerating {model_name} probabilities..."
    )

    all_probabilities = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    probabilities = np.concatenate(
        all_probabilities,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    print(
        f"{model_name} probability shape:",
        probabilities.shape
    )

    return probabilities, labels


# ============================================================
# FUNCTION: DINOv2 PROBABILITIES
# ============================================================
def get_dino_probabilities(dataset, dataset_name):

    print(
        f"\nGenerating DINOv2 scores for {dataset_name}..."
    )

    image_paths = []
    labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = os.path.join(dataset, class_name)

        for filename in sorted(os.listdir(class_dir)):

            if filename.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                image_paths.append(
                    os.path.join(class_dir, filename)
                )
                labels.append(class_index)

    all_features = []
    total_images = len(image_paths)

    from PIL import Image

    for start in range(0, total_images, BATCH_SIZE):

        batch_paths = image_paths[
            start:start + BATCH_SIZE
        ]

        batch_images = []

        for image_path in batch_paths:
            image = Image.open(image_path).convert("RGB")
            image = dino_transform(image)
            batch_images.append(image)

        batch_tensor = torch.stack(batch_images).to(DEVICE)

        with torch.no_grad():
            features = dinov2(batch_tensor)

        all_features.append(features.cpu().numpy())

        processed = min(start + BATCH_SIZE, total_images)

        print(
            f"{dataset_name}: {processed}/{total_images}"
        )

    features = np.concatenate(all_features, axis=0)
    labels = np.array(labels)

    # Apply the saved StandardScaler.
    scaler = dino_pipeline.named_steps["scaler"]
    svm = dino_pipeline.named_steps["svm"]

    scaled_features = scaler.transform(features)

    # Obtain SVM decision scores.
    decision_scores = svm.decision_function(scaled_features)

    # Convert multiclass decision scores into normalized
    # positive values for approximate soft voting.
    decision_scores = np.asarray(decision_scores)

    if decision_scores.ndim == 1:
        raise ValueError(
            "Expected multiclass SVM decision scores."
        )

    # Align the score columns with CLASS_NAMES.
    svm_classes = list(svm.classes_)
    class_indices = [svm_classes.index(i) for i in range(len(CLASS_NAMES))]

    decision_scores = decision_scores[:, class_indices]

    # Stable softmax conversion.
    decision_scores = decision_scores - np.max(
        decision_scores, axis=1, keepdims=True
    )

    exp_scores = np.exp(decision_scores)

    probabilities = exp_scores / np.sum(
        exp_scores, axis=1, keepdims=True
    )

    print(
        "DINOv2 score shape:",
        probabilities.shape
    )

    return probabilities, labels



# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("STEP 1: VALIDATION PREDICTIONS")
print("=" * 75)

resnet_val_prob, y_val = get_cnn_probabilities(
    resnet18,
    val_loader,
    "ResNet18"
)

efficient_val_prob, y_val_eff = get_cnn_probabilities(
    efficientnet,
    val_loader,
    "EfficientNet-B0"
)

dino_val_prob, y_val_dino = get_dino_probabilities(
    VAL_DIR,
    "Validation"
)

# Verify labels match.

assert np.array_equal(
    y_val,
    y_val_eff
)

assert np.array_equal(
    y_val,
    y_val_dino
)

print("\nValidation labels verified.")

# ============================================================
# VALIDATION INDIVIDUAL MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 75)
print("INDIVIDUAL VALIDATION PERFORMANCE")
print("=" * 75)

resnet_val_pred = np.argmax(
    resnet_val_prob,
    axis=1
)

efficient_val_pred = np.argmax(
    efficient_val_prob,
    axis=1
)

dino_val_pred = np.argmax(
    dino_val_prob,
    axis=1
)

print(
    "ResNet18:",
    f"{accuracy_score(y_val, resnet_val_pred) * 100:.2f}%"
)

print(
    "EfficientNet-B0:",
    f"{accuracy_score(y_val, efficient_val_pred) * 100:.2f}%"
)

print(
    "DINOv2 + SVM:",
    f"{accuracy_score(y_val, dino_val_pred) * 100:.2f}%"
)

# ============================================================
# SEARCH ENSEMBLE WEIGHTS
# ============================================================

print("\n" + "=" * 75)
print("STEP 2: SEARCHING ENSEMBLE WEIGHTS")
print("=" * 75)

best_accuracy = 0.0
best_macro_f1 = 0.0
best_weights = None

results_grid = []

# Search weights in 0.05 increments.
#
# ResNet weight     = w1
# EfficientNet      = w2
# DINOv2            = w3
#
# w1 + w2 + w3 = 1

weight_values = np.arange(
    0.0,
    1.01,
    0.05
)

for w1 in weight_values:

    for w2 in weight_values:

        w3 = 1.0 - w1 - w2

        if w3 < -1e-9:
            continue

        if w3 > 1.0:
            continue

        # Avoid floating-point issues.
        w3 = round(
            float(w3),
            2
        )

        ensemble_prob = (
            w1 * resnet_val_prob
            +
            w2 * efficient_val_prob
            +
            w3 * dino_val_prob
        )

        ensemble_pred = np.argmax(
            ensemble_prob,
            axis=1
        )

        acc = accuracy_score(
            y_val,
            ensemble_pred
        )

        macro_f1 = f1_score(
            y_val,
            ensemble_pred,
            average="macro"
        )

        results_grid.append({
            "resnet_weight": float(w1),
            "efficientnet_weight": float(w2),
            "dinov2_weight": float(w3),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1)
        })

        # Primary selection: validation Macro F1.
        # Accuracy is used as a tie-breaker.

        if (
            macro_f1 > best_macro_f1
            or (
                abs(macro_f1 - best_macro_f1) < 1e-12
                and acc > best_accuracy
            )
        ):

            best_macro_f1 = macro_f1
            best_accuracy = acc

            best_weights = (
                float(w1),
                float(w2),
                float(w3)
            )

# ============================================================
# BEST VALIDATION WEIGHTS
# ============================================================

print("\n" + "=" * 75)
print("BEST VALIDATION ENSEMBLE")
print("=" * 75)

print(
    "ResNet18 weight:",
    best_weights[0]
)

print(
    "EfficientNet-B0 weight:",
    best_weights[1]
)

print(
    "DINOv2 weight:",
    best_weights[2]
)

print(
    f"Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Validation Macro F1: "
    f"{best_macro_f1 * 100:.2f}%"
)

# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 75)
print("STEP 3: CLEAN TEST PREDICTIONS")
print("=" * 75)

resnet_test_prob, y_test = get_cnn_probabilities(
    resnet18,
    test_loader,
    "ResNet18"
)

efficient_test_prob, y_test_eff = get_cnn_probabilities(
    efficientnet,
    test_loader,
    "EfficientNet-B0"
)

dino_test_prob, y_test_dino = get_dino_probabilities(
    TEST_DIR,
    "Clean Test"
)

# Verify test labels.

assert np.array_equal(
    y_test,
    y_test_eff
)

assert np.array_equal(
    y_test,
    y_test_dino
)

print("\nTest labels verified.")

# ============================================================
# INDIVIDUAL TEST PREDICTIONS
# ============================================================

resnet_test_pred = np.argmax(
    resnet_test_prob,
    axis=1
)

efficient_test_pred = np.argmax(
    efficient_test_prob,
    axis=1
)

dino_test_pred = np.argmax(
    dino_test_prob,
    axis=1
)

# ============================================================
# APPLY BEST VALIDATION WEIGHTS
# ============================================================

w_resnet = best_weights[0]
w_efficient = best_weights[1]
w_dino = best_weights[2]

ensemble_test_prob = (
    w_resnet * resnet_test_prob
    +
    w_efficient * efficient_test_prob
    +
    w_dino * dino_test_prob
)

ensemble_test_pred = np.argmax(
    ensemble_test_prob,
    axis=1
)

# ============================================================
# TEST METRICS
# ============================================================

resnet_test_accuracy = accuracy_score(
    y_test,
    resnet_test_pred
)

resnet_test_f1 = f1_score(
    y_test,
    resnet_test_pred,
    average="macro"
)

efficient_test_accuracy = accuracy_score(
    y_test,
    efficient_test_pred
)

efficient_test_f1 = f1_score(
    y_test,
    efficient_test_pred,
    average="macro"
)

dino_test_accuracy = accuracy_score(
    y_test,
    dino_test_pred
)

dino_test_f1 = f1_score(
    y_test,
    dino_test_pred,
    average="macro"
)

ensemble_test_accuracy = accuracy_score(
    y_test,
    ensemble_test_pred
)

ensemble_test_f1 = f1_score(
    y_test,
    ensemble_test_pred,
    average="macro"
)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("FINAL CLEAN TEST RESULTS")
print("=" * 75)

print(
    f"ResNet18:        "
    f"{resnet_test_accuracy * 100:.2f}% "
    f"Accuracy | "
    f"{resnet_test_f1 * 100:.2f}% Macro F1"
)

print(
    f"EfficientNet-B0: "
    f"{efficient_test_accuracy * 100:.2f}% "
    f"Accuracy | "
    f"{efficient_test_f1 * 100:.2f}% Macro F1"
)

print(
    f"DINOv2 + SVM:    "
    f"{dino_test_accuracy * 100:.2f}% "
    f"Accuracy | "
    f"{dino_test_f1 * 100:.2f}% Macro F1"
)

print(
    f"ENSEMBLE:        "
    f"{ensemble_test_accuracy * 100:.2f}% "
    f"Accuracy | "
    f"{ensemble_test_f1 * 100:.2f}% Macro F1"
)

# ============================================================
# ENSEMBLE CLASSIFICATION REPORT
# ============================================================

print("\nEnsemble Classification Report:")

print(
    classification_report(
        y_test,
        ensemble_test_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
)

# ============================================================
# ENSEMBLE CONFUSION MATRIX
# ============================================================

ensemble_cm = confusion_matrix(
    y_test,
    ensemble_test_pred
)

print("\nEnsemble Confusion Matrix:")

print(
    ensemble_cm
)

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "test_images": int(len(y_test)),

    "best_validation_weights": {
        "resnet18": w_resnet,
        "efficientnet_b0": w_efficient,
        "dinov2_svm": w_dino
    },

    "validation": {
        "accuracy": float(best_accuracy),
        "macro_f1": float(best_macro_f1)
    },

    "test": {
        "resnet18": {
            "accuracy": float(resnet_test_accuracy),
            "macro_f1": float(resnet_test_f1)
        },

        "efficientnet_b0": {
            "accuracy": float(efficient_test_accuracy),
            "macro_f1": float(efficient_test_f1)
        },

        "dinov2_svm": {
            "accuracy": float(dino_test_accuracy),
            "macro_f1": float(dino_test_f1)
        },

        "ensemble": {
            "accuracy": float(ensemble_test_accuracy),
            "macro_f1": float(ensemble_test_f1),
            "confusion_matrix": ensemble_cm.tolist()
        }
    },

    "weight_search_count": len(
        results_grid
    ),

    "weight_search": results_grid
}

with open(
    RESULTS_PATH,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\nResults saved to:")
print(RESULTS_PATH)

print("\n" + "=" * 75)
print("ENSEMBLE EXPERIMENT COMPLETE")
print("=" * 75)