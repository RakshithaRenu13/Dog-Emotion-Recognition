
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = (
    "runs/classify/runs/dog_emotion/"
    "yolo11n_cls/weights/best.pt"
)
TEST_DIR = Path("Dataset_test")
OUTPUT_PATH = "confusion_matrix_yolo11.png"
BATCH_SIZE = 32

classes = ["angry", "curious", "happy", "sad", "sleepy"]
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Load model
model = YOLO(MODEL_PATH)
print("YOLO11 model loaded successfully.")

# Check dataset
if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}")

y_true = []
image_paths = []

for class_name in classes:
    folder = TEST_DIR / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing class folder: {folder}")

    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    image_paths.extend(files)
    y_true.extend([classes.index(class_name)] * len(files))

if not image_paths:
    raise ValueError("No test images found.")

print("Test images:", len(image_paths))

# Predict in batches
y_pred = []

for start in range(0, len(image_paths), BATCH_SIZE):
    batch_paths = image_paths[start:start + BATCH_SIZE]
    results = model.predict(
        source=[str(p) for p in batch_paths],
        imgsz=224,
        batch=BATCH_SIZE,
        verbose=False
    )

    for result in results:
        predicted_name = result.names[int(result.probs.top1)]
        if predicted_name not in classes:
            raise ValueError(f"Unexpected predicted class: {predicted_name}")
        y_pred.append(classes.index(predicted_name))

# Evaluation
accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 55)
print(f"YOLO11 Test Accuracy: {accuracy * 100:.2f}%")
print("=" * 55)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(classes))),
    target_names=classes,
    digits=4,
    zero_division=0
))

# Confusion matrix
cm = confusion_matrix(
    y_true,
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
plt.title("YOLO11 - Test Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.close()

print("\nConfusion matrix saved to:", OUTPUT_PATH)