import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)
import joblib
import json
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# IMPORTANT:
# Use the completely unseen test dataset.
TEST_DIR = os.path.join(
    BASE_DIR, "Dataset_test"
)

RESNET_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet18_emotion_classifier.pth"
)

EFFICIENTNET_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnet_b0_emotion_classifier.pth"
)

DINOV2_SVM_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dinov2_svm_tuned.joblib"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
NUM_CLASSES = 5

CLASS_NAMES = [
    "angry",
    "curious",
    "happy",
    "sad",
    "sleepy"
]

# Same weights used in your validation ensemble
WEIGHTS = [
    1 / 3,
    1 / 3,
    1 / 3
]

print("=" * 60)
print("DOG EMOTION RECOGNITION - ENSEMBLE TEST")
print("=" * 60)

print("Device:", DEVICE)
print("Test dataset:", TEST_DIR)


# ============================================================
# 2. TEST DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

print("\nTest classes:", test_dataset.classes)
print("Number of test images:", len(test_dataset))

if test_dataset.classes != CLASS_NAMES:
    raise ValueError(
        f"Class order mismatch.\n"
        f"Found: {test_dataset.classes}\n"
        f"Expected: {CLASS_NAMES}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)


# ============================================================
# 3. LOAD CHECKPOINT
# ============================================================

def load_state(path):

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    if isinstance(checkpoint, dict):

        for key in (
            "state_dict",
            "model_state_dict",
            "model"
        ):

            if (
                key in checkpoint
                and isinstance(checkpoint[key], dict)
            ):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported checkpoint format: {path}"
        )

    cleaned = {}

    for key, value in checkpoint.items():

        key = key.removeprefix("module.")

        cleaned[key] = value

    return cleaned


# ============================================================
# 4. LOAD RESNET18
# ============================================================

print("\nLoading ResNet18...")

resnet = models.resnet18(weights=None)

resnet.fc = nn.Linear(
    resnet.fc.in_features,
    NUM_CLASSES
)

resnet.load_state_dict(
    load_state(RESNET_PATH),
    strict=True
)

resnet = resnet.to(DEVICE)
resnet.eval()

print("ResNet18 loaded.")


# ============================================================
# 5. LOAD EFFICIENTNET-B0
# ============================================================

print("\nLoading EfficientNet-B0...")

efficientnet = models.efficientnet_b0(
    weights=None
)

efficientnet.classifier[1] = nn.Linear(
    efficientnet.classifier[1].in_features,
    NUM_CLASSES
)

efficientnet.load_state_dict(
    load_state(EFFICIENTNET_PATH),
    strict=True
)

efficientnet = efficientnet.to(DEVICE)
efficientnet.eval()

print("EfficientNet-B0 loaded.")


# ============================================================
# 6. LOAD DINOv2 + SVM
# ============================================================

print("\nLoading DINOv2 + SVM...")

svm_data = joblib.load(
    DINOV2_SVM_PATH
)

svm = svm_data["svm_pipeline"]

dinov2_name = svm_data.get(
    "dinov2_name",
    "dinov2_vits14"
)

if dinov2_name != "dinov2_vits14":

    raise ValueError(
        f"Unexpected DINOv2 architecture: "
        f"{dinov2_name}"
    )

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

dinov2 = dinov2.to(DEVICE)
dinov2.eval()

print("DINOv2 loaded.")
print("SVM loaded.")


# ============================================================
# 7. GENERATE PREDICTIONS
# ============================================================

resnet_probs = []
efficientnet_probs = []
dinov2_probs = []

true_labels = []


@torch.inference_mode()
def predict_all():

    for images, labels in test_loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # ResNet18
        # ----------------------------------------------------

        r_logits = resnet(images)

        r_probs = F.softmax(
            r_logits,
            dim=1
        )

        resnet_probs.append(
            r_probs.cpu().numpy()
        )


        # ----------------------------------------------------
        # EfficientNet-B0
        # ----------------------------------------------------

        e_logits = efficientnet(images)

        e_probs = F.softmax(
            e_logits,
            dim=1
        )

        efficientnet_probs.append(
            e_probs.cpu().numpy()
        )


        # ----------------------------------------------------
        # DINOv2
        # ----------------------------------------------------

        features = dinov2.forward_features(
            images
        )

        embeddings = features[
            "x_norm_clstoken"
        ]

        embeddings = embeddings.cpu().numpy()


        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        decision = svm.decision_function(
            embeddings
        )

        if decision.ndim == 1:

            decision = np.column_stack(
                [-decision, decision]
            )


        # Convert SVM decision scores
        # to probability-like values

        decision = (
            decision
            - decision.max(
                axis=1,
                keepdims=True
            )
        )

        exp_scores = np.exp(decision)

        d_probs = (
            exp_scores
            / exp_scores.sum(
                axis=1,
                keepdims=True
            )
        )


        # ----------------------------------------------------
        # Match SVM class order
        # ----------------------------------------------------

        svm_classes = list(
            svm.classes_
        )

        if set(svm_classes) != set(
            range(NUM_CLASSES)
        ):

            raise ValueError(
                f"Unexpected SVM class labels: "
                f"{svm_classes}"
            )

        d_probs = d_probs[
            :,
            svm_classes
        ]

        dinov2_probs.append(
            d_probs
        )

        true_labels.extend(
            labels.numpy()
        )


print("\nRunning test prediction...")

predict_all()


# ============================================================
# 8. COMBINE PREDICTIONS
# ============================================================

resnet_probs = np.concatenate(
    resnet_probs
)

efficientnet_probs = np.concatenate(
    efficientnet_probs
)

dinov2_probs = np.concatenate(
    dinov2_probs
)

true_labels = np.array(
    true_labels
)

print("\nPrediction arrays created.")

print("ResNet18 shape:",
      resnet_probs.shape)

print("EfficientNet shape:",
      efficientnet_probs.shape)

print("DINOv2 shape:",
      dinov2_probs.shape)


# ============================================================
# 9. INDIVIDUAL MODEL RESULTS
# ============================================================

resnet_predictions = np.argmax(
    resnet_probs,
    axis=1
)

efficientnet_predictions = np.argmax(
    efficientnet_probs,
    axis=1
)

dinov2_predictions = np.argmax(
    dinov2_probs,
    axis=1
)


print("\n" + "=" * 60)
print("INDIVIDUAL TEST RESULTS")
print("=" * 60)

for name, predictions in [

    ("ResNet18", resnet_predictions),

    (
        "EfficientNet-B0",
        efficientnet_predictions
    ),

    (
        "DINOv2 + SVM",
        dinov2_predictions
    )

]:

    acc = accuracy_score(
        true_labels,
        predictions
    )

    f1 = f1_score(
        true_labels,
        predictions,
        average="macro"
    )

    print(
        f"{name}: "
        f"Accuracy={acc * 100:.2f}%, "
        f"Macro F1={f1 * 100:.2f}%"
    )


# ============================================================
# 10. SOFT-VOTING ENSEMBLE
# ============================================================

ensemble_probs = (

    WEIGHTS[0] * resnet_probs

    + WEIGHTS[1] * efficientnet_probs

    + WEIGHTS[2] * dinov2_probs

)

predictions = np.argmax(
    ensemble_probs,
    axis=1
)


# ============================================================
# 11. TEST METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

macro_f1 = f1_score(
    true_labels,
    predictions,
    average="macro"
)

cm = confusion_matrix(
    true_labels,
    predictions
)

report_text = classification_report(
    true_labels,
    predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

report_dict = classification_report(
    true_labels,
    predictions,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# ============================================================
# 12. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ENSEMBLE TEST RESULTS")
print("=" * 60)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{macro_f1 * 100:.2f}%"
)

print("\nClassification Report:")
print(report_text)

print("Confusion Matrix:")
print(cm)


# ============================================================
# 13. SAVE JSON RESULTS
# ============================================================

results = {

    "model_name":
        "Soft Voting Ensemble",

    "dataset":
        "Dataset_test",

    "class_names":
        CLASS_NAMES,

    "weights":
        WEIGHTS,

    "test_samples":
        int(len(test_dataset)),

    "test_accuracy":
        float(accuracy),

    "test_macro_f1":
        float(macro_f1),

    "confusion_matrix":
        cm.tolist(),

    "classification_report":
        report_dict
}

json_path = os.path.join(
    OUTPUT_DIR,
    "ensemble_test_results.json"
)

with open(
    json_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# 14. SAVE PREDICTIONS
# ============================================================

npz_path = os.path.join(
    OUTPUT_DIR,
    "ensemble_test_predictions.npz"
)

np.savez_compressed(
    npz_path,
    labels=true_labels,
    probabilities=ensemble_probs,
    predictions=predictions
)


# ============================================================
# 15. SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.imshow(cm)

plt.title(
    "Ensemble Test Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(NUM_CLASSES),
    CLASS_NAMES,
    rotation=45
)

plt.yticks(
    range(NUM_CLASSES),
    CLASS_NAMES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# Add numbers to matrix

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "ensemble_test_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. FINISHED
# ============================================================

print("\n" + "=" * 60)
print("FILES SAVED")
print("=" * 60)

print(
    "Results:",
    json_path
)

print(
    "Predictions:",
    npz_path
)

print(
    "Confusion matrix:",
    cm_path
)

print("\nTest evaluation completed successfully.")