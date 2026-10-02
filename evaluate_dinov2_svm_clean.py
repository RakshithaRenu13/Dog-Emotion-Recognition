import os
import json
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(
    BASE_DIR,
    "Dataset_test_clean"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dinov2_svm_tuned.joblib"
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

# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("DINOV2 + SVM CLEAN TEST EVALUATION")
print("=" * 70)

print("Device:", DEVICE)
print("Test directory:", TEST_DIR)
print("Model:", MODEL_PATH)

# ============================================================
# LOAD DINOv2
# ============================================================

print("\nLoading DINOv2...")

dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14"
)

dinov2 = dinov2.to(DEVICE)
dinov2.eval()

print("DINOv2 loaded successfully.")

# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD SAVED MODEL
# ============================================================

print("\nLoading saved DINOv2 + SVM model...")

saved_model = joblib.load(
    MODEL_PATH
)

print("Saved object type:", type(saved_model))

# The .joblib file is a dictionary.
# Extract the trained preprocessing + SVM pipeline.

svm_pipeline = saved_model["svm_pipeline"]

print("\nModel name:")
print(saved_model["model_name"])

print("\nDINOv2 model:")
print(saved_model["dinov2_name"])

print("\nImage size:")
print(saved_model["image_size"])

print("\nSaved validation accuracy:")
print(
    f"{saved_model['validation_accuracy'] * 100:.2f}%"
)

print("\nSaved validation Macro F1:")
print(
    f"{saved_model['validation_macro_f1'] * 100:.2f}%"
)

print("\nSVM pipeline:")
print(svm_pipeline)

# ============================================================
# LOAD TEST IMAGES
# ============================================================

image_paths = []
labels = []

for class_index, class_name in enumerate(CLASS_NAMES):

    class_dir = os.path.join(
        TEST_DIR,
        class_name
    )

    if not os.path.exists(class_dir):

        print(
            f"WARNING: Missing directory: {class_dir}"
        )

        continue

    for filename in sorted(
        os.listdir(class_dir)
    ):

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        ):

            image_paths.append(
                os.path.join(
                    class_dir,
                    filename
                )
            )

            labels.append(
                class_index
            )

print("\nTotal test images:", len(image_paths))

# ============================================================
# FEATURE EXTRACTION
# ============================================================

print("\nExtracting DINOv2 features...")

all_features = []

BATCH_SIZE = 32

for start in range(
    0,
    len(image_paths),
    BATCH_SIZE
):

    batch_paths = image_paths[
        start:start + BATCH_SIZE
    ]

    batch_images = []

    for image_path in batch_paths:

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(image)

        batch_images.append(
            image
        )

    batch_tensor = torch.stack(
        batch_images
    ).to(DEVICE)

    with torch.no_grad():

        features = dinov2.forward_features(
            batch_tensor
        )["x_norm_clstoken"]


            
        

    features = features.cpu().numpy()

    all_features.append(
        features
    )

    processed = min(
        start + BATCH_SIZE,
        len(image_paths)
    )

    print(
        f"Processed {processed}/{len(image_paths)}"
    )

# ============================================================
# COMBINE FEATURES
# ============================================================

X_test = np.concatenate(
    all_features,
    axis=0
)

y_test = np.array(
    labels
)

print("\nFeature shape:", X_test.shape)
print("Label shape:", y_test.shape)

# ============================================================
# VERIFY FEATURE DIMENSION
# ============================================================

print("\nChecking feature dimension...")

print(
    "DINOv2 feature dimension:",
    X_test.shape[1]
)

if hasattr(
    svm_pipeline,
    "n_features_in_"
):

    print(
        "Pipeline expected features:",
        svm_pipeline.n_features_in_
    )

# ============================================================
# SVM PREDICTION
# ============================================================

print("\nRunning saved StandardScaler + SVM...")

# IMPORTANT:
# svm_pipeline contains:
#
# StandardScaler
#       ↓
# SVM
#
# Therefore we must call predict()
# on the entire pipeline.

y_pred = svm_pipeline.predict(
    X_test
)

# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\n" + "=" * 70)
print("DINOV2 + SVM TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{macro_f1 * 100:.2f}%"
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

report = classification_report(
    y_test,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print(report)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)

# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "DINOv2 + SVM",
    "test_images": len(y_test),
    "accuracy": float(accuracy),
    "macro_f1": float(macro_f1),
    "classification_report": classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True
    ),
    "confusion_matrix": cm.tolist()
}

results_path = os.path.join(
    BASE_DIR,
    "models",
    "dinov2_svm_clean_test_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print("\nResults saved to:")
print(results_path)

print("\nEvaluation completed.")