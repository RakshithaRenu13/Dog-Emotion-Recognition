
"""
Dog Emotion Recognition with ResNet18 and Grad-CAM.
Trains once, saves the model, evaluates on a separate test set,
and generates Grad-CAM visualizations for misclassified images.
"""

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import src.config
from src.utils import DogEmotionDataset

# Configuration
random.seed(src.config.RANDOM_STATE)
torch.manual_seed(src.config.RANDOM_STATE)

DEVICE = src.config.DEVICE
CLASSES = src.config.EMOTION_CLASSES

# Change these paths if your folder names differ.
TRAIN_DIR = Path(src.config.DATASET_PATH)
TEST_DIR = Path("Dataset_segmented_test")

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "gradcam_resnet18.pth"
OUTPUT_DIR = Path("output/gradcam_failures")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_model():
    """Create the ResNet18 emotion classifier."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(DEVICE)


def train_model(train_loader, epochs=8):
    """Train ResNet18 on the training dataset."""
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)

        print(
            f"Epoch [{epoch + 1}/ {epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Acc: {accuracy:.2f}%"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH.resolve()}")
    return model


def load_or_train_model(train_loader):
    """Load the saved model or train it if it does not exist."""
    model = create_model()

    if MODEL_PATH.is_file():
        print(f"Loading saved model: {MODEL_PATH}")
        state = torch.load(
            MODEL_PATH, map_location=DEVICE, weights_only=True
        )
        model.load_state_dict(state)
        return model

    print("No saved Grad-CAM model found. Training ResNet18...")
    return train_model(train_loader)


def find_failures(model, test_loader):
    """Find misclassified test images and their predictions."""
    import torch.nn.functional as F

    model.eval()
    failures = []
    correct = 0
    total = 0

    # Grad-CAM is not calculated here, so inference can use no_grad.
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            for i in range(labels.size(0)):
                total += 1
                pred = predictions[i].item()
                true = labels[i].item()

                if pred == true:
                    correct += 1
                else:
                    failures.append({
                        "image": images[i].cpu(),
                        "true_label": true,
                        "pred_label": pred,
                        "confidence": probabilities[i, pred].item(),
                        "path": str(paths[i])
                    })

    accuracy = 100.0 * correct / max(total, 1)
    print(f"Test samples: {total}")
    print(f"Test accuracy: {accuracy:.2f}%")
    print(f"Misclassified samples: {len(failures)}")

    return failures


def denormalize(tensor):
    """Convert a normalized image tensor to an RGB image."""
    mean = torch.tensor(
        src.config.IMAGENET_MEAN
    ).view(3, 1, 1)

    std = torch.tensor(
        src.config.IMAGENET_STD
    ).view(3, 1, 1)

    image = tensor.cpu() * std + mean
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def generate_visualizations(model, failures, limit=10):
    """Generate and save Grad-CAM for selected failures."""
    if not failures:
        print("No misclassified images found.")
        return

    # Visualize the most confident incorrect predictions first.
    failures.sort(key=lambda item: item["confidence"], reverse=True)
    selected = failures[:limit]

    target_layer = model.layer4[-1]

    model.eval()
    print(f"Generating Grad-CAM for {len(selected)} failures...")

    # GradCAM registers hooks on the model. Close it when finished.
    with GradCAM(
        model=model,
        target_layers=[target_layer]
    ) as cam:

        for i, failure in enumerate(selected, start=1):
            image_tensor = failure["image"].unsqueeze(0).to(DEVICE)

            # Grad-CAM requires gradients. Do not use torch.no_grad here.
            target = [
                ClassifierOutputTarget(failure["pred_label"])
            ]
            grayscale_cam = cam(
                input_tensor=image_tensor,
                targets=target
            )[0]

            rgb_image = denormalize(failure["image"])
            visualization = show_cam_on_image(
                rgb_image,
                grayscale_cam,
                use_rgb=True
            )

            true_name = CLASSES[failure["true_label"]]
            pred_name = CLASSES[failure["pred_label"]]
            confidence = failure["confidence"] * 100

            filename = (
                f"failure_{i}_{true_name}_as_{pred_name}.jpg"
            )
            save_path = OUTPUT_DIR / filename

            cv2.imwrite(
                str(save_path),
                cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            )

            print(
                f"Saved: {filename} | "
                f"Actual: {true_name} | "
                f"Predicted: {pred_name} | "
                f"Confidence: {confidence:.2f}%"
            )


def main():
    print("=" * 60)
    print("DOG EMOTION RECOGNITION - GRAD-CAM")
    print("=" * 60)
    print("Device:", DEVICE)

    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(src.config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            src.config.IMAGENET_MEAN,
            src.config.IMAGENET_STD
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((
            src.config.IMAGE_SIZE,
            src.config.IMAGE_SIZE
        )),
        transforms.ToTensor(),
        transforms.Normalize(
            src.config.IMAGENET_MEAN,
            src.config.IMAGENET_STD
        )
    ])

    print("\n[1/4] Loading datasets...")
    train_dataset = DogEmotionDataset(
        TRAIN_DIR, transform=train_transform
    )
    test_dataset = DogEmotionDataset(
        TEST_DIR, transform=test_transform
    )

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("Training or test dataset is empty.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=src.config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Training samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))

    print("\n[2/4] Loading or training model...")
    model = load_or_train_model(train_loader)

    print("\n[3/4] Evaluating on the test set...")
    failures = find_failures(model, test_loader)

    print("\n[4/4] Generating Grad-CAM visualizations...")
    generate_visualizations(model, failures, limit=10)

    print("\nCompleted!")
    print("Grad-CAM images:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()