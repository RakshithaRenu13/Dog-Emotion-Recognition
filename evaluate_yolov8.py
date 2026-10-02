
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np

BASE_DIR = Path(r"C:\Users\acer\dogs-emotion-recognition")
MODEL_PATH = BASE_DIR / "models" / "yolo_emotion_classifier.pt"
TEST_DIR = BASE_DIR / "Dataset_test"

class_names = ["angry", "curious", "happy", "sad", "sleepy"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

model = YOLO(str(MODEL_PATH))

y_true = []
y_pred = []

for label, emotion in enumerate(class_names):
    folder = TEST_DIR / emotion
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing class folder: {folder}")

    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }
    )

    for start in range(0, len(images), 32):
        batch_paths = images[start:start + 32]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=224,
            device=0,
            verbose=False
        )

        for result in results:
            predicted_name = result.names[result.probs.top1]
            if predicted_name not in class_names:
                raise ValueError(
                    f"Unexpected model class: {predicted_name}"
                )

            y_true.append(label)
            y_pred.append(class_names.index(predicted_name))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print("\n" + "=" * 60)
print("YOLOv8 CLEAN TEST RESULTS")
print("=" * 60)
print(f"Test images: {len(y_true)}")
print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(
    f"Macro F1: "
    f"{f1_score(y_true, y_pred, average='macro') * 100:.2f}%"
)

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    digits=4,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    y_true, y_pred, labels=list(range(len(class_names)))
))