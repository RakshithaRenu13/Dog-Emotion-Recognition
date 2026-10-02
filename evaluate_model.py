
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import EMOTION_CLASSES
from src.utils.data_loader import DogEmotionDataset


def create_model(model_name, num_classes):
    """Create the selected model architecture."""

    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(
            model.fc.in_features, num_classes
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )

    else:
        raise ValueError(
            f"Unsupported evaluation model: {model_name}"
        )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a dog emotion classification model"
    )

    parser.add_argument(
        "--model",
        choices=["resnet18", "efficientnet_b0"],
        default="resnet18",
        help="Model architecture"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint"
    )

    parser.add_argument(
        "--test-dir",
        type=str,
        default="Dataset_test",
        help="Independent test dataset directory"
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    test_dir = Path(args.test_dir)

    if args.checkpoint:
        model_path = Path(args.checkpoint)
    else:
        model_path = Path(
            f"models/{args.model}_emotion_classifier.pth"
        )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_dir}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    print("\nTest dataset:", test_dir)
    print("Model:", args.model)
    print("Checkpoint:", model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_dataset = DogEmotionDataset(
        root_path=test_dir,
        transform=transform
    )

    if len(test_dataset) == 0:
        raise ValueError("No test images found.")

    print("Test images:", len(test_dataset))
    print("Classes:", EMOTION_CLASSES)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print("\nLoading checkpoint...")

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    saved_model_name = checkpoint.get("model_name")

    if saved_model_name:
        print("Saved architecture:", saved_model_name)
        if saved_model_name != args.model:
            raise ValueError(
                f"Checkpoint is for {saved_model_name}, "
                f"but --model is {args.model}."
            )

    saved_classes = checkpoint.get("class_names")

    if saved_classes is not None:
        print("Saved classes:", saved_classes)
        if list(saved_classes) != list(EMOTION_CLASSES):
            raise ValueError(
                "Checkpoint class order does not match "
                "the current emotion class order."
            )

    model = create_model(
        args.model,
        len(EMOTION_CLASSES)
    )

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    print("\nRunning evaluation...")

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print(f"Model: {args.model}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(EMOTION_CLASSES))),
        target_names=EMOTION_CLASSES,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(EMOTION_CLASSES)))
    )

    print("\nConfusion Matrix:")
    print(cm)

    print("\nPer-Class Accuracy:")
    for i, emotion in enumerate(EMOTION_CLASSES):
        total = cm[i].sum()
        correct = cm[i, i]
        class_accuracy = (
            correct / total * 100 if total > 0 else 0.0
        )
        print(
            f"{emotion:10s}: {class_accuracy:.2f}% "
            f"({correct}/{total})"
        )

    output_path = Path(
        f"confusion_matrix_{args.model}.png"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=EMOTION_CLASSES,
        yticklabels=EMOTION_CLASSES
    )

    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.title(
        f"{args.model} Dog Emotion Confusion Matrix"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nConfusion matrix saved as:", output_path)


if __name__ == "__main__":
    main()