
from pathlib import Path
import numpy as np
import torch
import joblib

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

# Configuration
TRAIN_DIR = "datasets/dog_emotions/train"
VAL_DIR = "datasets/dog_emotions/val"
SAVE_PATH = "models/dinov2_svm_tuned.joblib"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_ds = datasets.ImageFolder(VAL_DIR, transform=transform)

if train_ds.classes != val_ds.classes:
    raise ValueError("Training and validation classes differ.")

classes = train_ds.classes

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

# Load frozen DINOv2 feature extractor
model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14",
    pretrained=True
).to(device)
model.eval()

def extract_features(loader):
    features_all = []
    labels_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.forward_features(images)["x_norm_clstoken"]
            features_all.append(features.cpu().numpy())
            labels_all.append(labels.numpy())

    return (
        np.concatenate(features_all),
        np.concatenate(labels_all)
    )

print("Extracting training features...")
X_train, y_train = extract_features(train_loader)

print("Extracting validation features...")
X_val, y_val = extract_features(val_loader)

# Hyperparameter tuning using training data only
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        class_weight="balanced"
    ))
])

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", "auto"]
}

search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
    verbose=2
)

print("\nTuning SVM...")
search.fit(X_train, y_train)

print("\nBest parameters:", search.best_params_)
print("Training CV macro F1:", search.best_score_)

# Evaluate selected configuration on validation data
best_svm = search.best_estimator_
predictions = best_svm.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
print(f"\nValidation accuracy: {accuracy * 100:.2f}%")
print("\nValidation classification report:")
print(classification_report(
    y_val,
    predictions,
    target_names=classes,
    digits=4,
    zero_division=0
))

# Save tuned model
Path(SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "model_name": "dinov2_vits14_svm_tuned",
    "class_names": classes,
    "svm_pipeline": best_svm,
    "validation_accuracy": accuracy,
    "validation_macro_f1": float(
        __import__("sklearn.metrics", fromlist=["f1_score"])
        .f1_score(y_val, predictions, average="macro")
    ),
    "dinov2_name": "dinov2_vits14",
    "image_size": 224,
    "best_params": search.best_params_
}, SAVE_PATH)

print("\nTuned model saved to:", SAVE_PATH)