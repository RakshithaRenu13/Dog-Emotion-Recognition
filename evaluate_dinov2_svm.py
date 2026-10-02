
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Configuration
TEST_DIR = "Dataset_test"
MODEL_PATH = "models/dinov2_svm_classifier.joblib"
OUTPUT_PATH = "confusion_matrix_dinov2_svm.png"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load saved model
saved = joblib.load(MODEL_PATH)
svm = saved["svm_pipeline"]
classes = list(saved["class_names"])
dinov2_name = saved["dinov2_name"]

print("Model:", saved["model_name"])
print("Classes:", classes)

# Load test dataset
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

if test_dataset.classes != classes:
    raise ValueError(
        f"Class mismatch!\n"
        f"Dataset: {test_dataset.classes}\n"
        f"Model: {classes}"
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# Load pretrained DINOv2
dinov2 = torch.hub.load(
    "facebookresearch/dinov2",
    dinov2_name,
    pretrained=True
)
dinov2 = dinov2.to(device)
dinov2.eval()

# Extract features
all_features = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        features = dinov2.forward_features(images)[
            "x_norm_clstoken"
        ]

        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

X_test = np.concatenate(all_features, axis=0)
y_test = np.concatenate(all_labels, axis=0)

print("\nTest images:", len(y_test))
print("Feature shape:", X_test.shape)

# Predict
y_pred = svm.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print(f"DINOv2 + SVM Test Accuracy: {accuracy * 100:.2f}%")
print("=" * 55)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    digits=4,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(len(classes)))
)

print("\nConfusion Matrix:")
print(cm)

# Save confusion matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    xticks_rotation=45
)

plt.title("DINOv2 + SVM - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print("\nConfusion matrix saved to:", OUTPUT_PATH)