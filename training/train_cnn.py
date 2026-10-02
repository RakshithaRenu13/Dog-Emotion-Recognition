"""
Train CNN classifiers for dog emotion recognition.

Example:

    python -m training.train_cnn --model resnet18 --epochs 25 --batch-size 32 --lr 0.0001
"""

import argparse

from pathlib import Path

from src.config import (
    DATASET_PATH,
    MODELS_DIR
)

from src.models.cnn_classifier import (
    CNNEmotionClassifier
)


def main():

    # ======================================================
    # ARGUMENTS
    # ======================================================

    parser = argparse.ArgumentParser(
        description=(
            "Train CNN for dog emotion classification"
        )
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=[
            "resnet50",
            "resnet18",
            "densenet121",
            "mobilenet_v2",
            "inception_v3",
            "efficientnet_b0",
            "efficientnet_b1",
            "vgg16",
            "convnext_tiny",
            "convnext_small"
        ],
        help="Model architecture"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.20,
        help="Validation split ratio"
    )

    args = parser.parse_args()

    # ======================================================
    # PROJECT INFORMATION
    # ======================================================

    print(
        "=" * 60
    )

    print(
        f"Dog Emotion Classification "
        f"with {args.model.upper()}"
    )

    print(
        "=" * 60
    )

    print(
        f"\nDataset: {DATASET_PATH}"
    )

    print(
        f"Epochs: {args.epochs}"
    )

    print(
        f"Batch size: {args.batch_size}"
    )

    print(
        f"Learning rate: {args.lr}"
    )

    print(
        f"Validation split: "
        f"{args.val_split * 100:.0f}%"
    )

    # ======================================================
    # CREATE MODELS DIRECTORY
    # ======================================================

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # INITIALIZE MODEL
    # ======================================================

    print(
        "\n[1/4] Initializing model..."
    )

    classifier = CNNEmotionClassifier(

        model_name=args.model,

        pretrained=True,

        # IMPORTANT:
        # False = fine-tune entire CNN
        freeze_features=False
    )

    # ======================================================
    # PREPARE DATA
    # ======================================================

    print(
        "\n[2/4] Preparing data..."
    )

    train_loader, val_loader = (
        classifier.prepare_data(

            data_dir=DATASET_PATH,

            batch_size=args.batch_size,

            val_split=args.val_split
        )
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Classes: "
        f"{classifier.class_names}"
    )

    # ======================================================
    # TRAIN
    # ======================================================

    print(
        "\n[3/4] Training model..."
    )

    best_accuracy = classifier.train(

        train_loader=train_loader,

        val_loader=val_loader,

        num_epochs=args.epochs,

        learning_rate=args.lr
    )

    # ======================================================
    # SAVE
    # ======================================================

    print(
        "\n[4/4] Saving best model..."
    )

    model_save_path = (
        MODELS_DIR
        / f"{args.model}_emotion_classifier.pth"
    )

    classifier.save(
        model_save_path
    )

    # ======================================================
    # FINAL INFORMATION
    # ======================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "Training complete!"
    )

    print(
        f"Best validation accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Model saved to: "
        f"{model_save_path}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()