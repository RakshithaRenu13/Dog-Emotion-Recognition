
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
from sklearn.preprocessing import normalize
import joblib
import json

# -------------------------------
# 1. CONFIGURATION
# -------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VAL_DIR = os.path.join(
    BASE_DIR, "datasets", "dog_emotions", "val"
)

# Change these two paths to your actual checkpoint files.
RESNET_PATH = os.path.join(
    BASE_DIR, "models", "resnet18_emotion_classifier.pth"
)
EFFICIENTNET_PATH = os.path.join(
    BASE_DIR, "models", "efficientnet_b0_emotion_classifier.pth"
)

DINOV2_SVM_PATH = os.path.join(
    BASE_DIR, "models", "dinov2_svm_tuned.joblib"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 5

CLASS_NAMES = ["angry", "curious", "happy", "sad", "sleepy"]

# Equal weights for all three models initially.
WEIGHTS = [1/3, 1/3, 1/3]

print("Device:", DEVICE)

# -------------------------------
# 2. VALIDATION DATASET
# -------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(VAL_DIR, transform=transform)

if dataset.classes != CLASS_NAMES:
    raise ValueError(
        f"Class order mismatch. Found {dataset.classes}, "
        f"expected {CLASS_NAMES}"
    )

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

# -------------------------------
# 3. LOAD CHECKPOINTS
# -------------------------------

def load_state(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Update the checkpoint path in ensemble.py."
        )

    checkpoint = torch.load(
        path, map_location=DEVICE, weights_only=False
    )

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    # Remove DataParallel prefixes, if present.
    cleaned = {}
    for key, value in checkpoint.items():
        key = key.removeprefix("module.")
        cleaned[key] = value

    return cleaned


# -------------------------------
# 4. RESNET18
# -------------------------------

resnet = models.resnet18(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)

resnet.load_state_dict(load_state(RESNET_PATH), strict=True)
resnet = resnet.to(DEVICE).eval()

# -------------------------------
# 5. EFFICIENTNET-B0
# -------------------------------

efficientnet = models.efficientnet_b0(weights=None)
efficientnet.classifier[1] = nn.Linear(
    efficientnet.classifier[1].in_features,
    NUM_CLASSES
)

efficientnet.load_state_dict(
    load_state(EFFICIENTNET_PATH), strict=True
)
efficientnet = efficientnet.to(DEVICE).eval()

# -------------------------------
# 6. DINOv2 + SVM
# -------------------------------

svm_data = joblib.load(DINOV2_SVM_PATH)

svm = svm_data["svm_pipeline"]

dinov2_name = svm_data.get(
    "dinov2_name", "dinov2_vits14"
)

if dinov2_name != "dinov2_vits14":
    raise ValueError(
        f"Unexpected DINOv2 architecture: {dinov2_name}"
    )

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
).to(DEVICE).eval()

# -------------------------------
# 7. PREDICTION
# -------------------------------

resnet_probs = []
efficientnet_probs = []
dinov2_probs = []
true_labels = []

@torch.inference_mode()
def predict_all():
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)

        # ResNet18 probabilities
        r_logits = resnet(images)
        r_probs = F.softmax(r_logits, dim=1)
        resnet_probs.append(r_probs.cpu().numpy())

        # EfficientNet-B0 probabilities
        e_logits = efficientnet(images)
        e_probs = F.softmax(e_logits, dim=1)
        efficientnet_probs.append(e_probs.cpu().numpy())

        # DINOv2 feature extraction
        features = dinov2.forward_features(images)
        embeddings = features["x_norm_clstoken"]
        embeddings = embeddings.cpu().numpy()

        # SVM decision scores converted to normalized
        # positive scores for probability-like voting.
        decision = svm.decision_function(embeddings)

        # Binary classifiers have a different output shape.
        if decision.ndim == 1:
            decision = np.column_stack([-decision, decision])

        # Softmax of the SVM decision scores.
        decision = decision - decision.max(axis=1, keepdims=True)
        exp_scores = np.exp(decision)
        d_probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        # Ensure class order matches ImageFolder.
        svm_classes = list(svm.classes_)
        if set(svm_classes) != set(range(NUM_CLASSES)):
            raise ValueError(
                f"Unexpected SVM class labels: {svm_classes}"
            )

        d_probs = d_probs[:, svm_classes]
        dinov2_probs.append(d_probs)

        true_labels.extend(labels.numpy())

predict_all()

resnet_probs = np.concatenate(resnet_probs)
efficientnet_probs = np.concatenate(efficientnet_probs)
dinov2_probs = np.concatenate(dinov2_probs)
true_labels = np.array(true_labels)

# -------------------------------
# 8. SOFT-VOTING ENSEMBLE
# -------------------------------

ensemble_probs = (
    WEIGHTS[0] * resnet_probs
    + WEIGHTS[1] * efficientnet_probs
    + WEIGHTS[2] * dinov2_probs
)

predictions = np.argmax(ensemble_probs, axis=1)

accuracy = accuracy_score(true_labels, predictions)
macro_f1 = f1_score(
    true_labels, predictions, average="macro"
)

print("\nIndividual validation results:")
for name, probs in [
    ("ResNet18", resnet_probs),
    ("EfficientNet-B0", efficientnet_probs),
    ("DINOv2 + SVM", dinov2_probs)
]:
    pred = np.argmax(probs, axis=1)
    print(
        f"{name}: Accuracy={accuracy_score(true_labels, pred)*100:.2f}%, "
        f"Macro F1={f1_score(true_labels, pred, average='macro')*100:.2f}%"
    )

print("\nENSEMBLE RESULTS")
print(f"Validation accuracy: {accuracy*100:.2f}%")
print(f"Validation macro F1: {macro_f1*100:.2f}%")

print("\nClassification report:")
print(classification_report(
    true_labels,
    predictions,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
))

print("\nConfusion matrix:")
print(confusion_matrix(true_labels, predictions))

# -------------------------------
# 9. SAVE RESULTS
# -------------------------------

results = {
    "model_name": "Soft Voting Ensemble",
    "class_names": CLASS_NAMES,
    "weights": WEIGHTS,
    "validation_accuracy": float(accuracy),
    "validation_macro_f1": float(macro_f1),
    "confusion_matrix": confusion_matrix(
        true_labels, predictions
    ).tolist(),
    "classification_report": classification_report(
        true_labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
}

output_path = os.path.join(
    OUTPUT_DIR, "ensemble_validation_results.json"
)

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "ensemble_validation_predictions.npz"),
    labels=true_labels,
    probabilities=ensemble_probs,
    predictions=predictions
)

print("\nResults saved to:", output_path)
print("Prediction probabilities saved.")