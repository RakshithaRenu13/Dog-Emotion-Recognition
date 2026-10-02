import os
import shutil
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

SOURCE = "Dataset_trainval_clean"

TRAIN_OUT = "datasets/dog_emotions_clean/train"
VAL_OUT = "datasets/dog_emotions_clean/val"

TRAIN_RATIO = 0.80
RANDOM_STATE = 42

print("=" * 60)
print("CREATING CLEAN TRAIN / VALIDATION SPLIT")
print("=" * 60)

dataset = ImageFolder(SOURCE)

print("Total images:", len(dataset))
print("Classes:", dataset.classes)

# Create output directories
if os.path.exists("datasets/dog_emotions_clean"):
    shutil.rmtree("datasets/dog_emotions_clean")

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(VAL_OUT, exist_ok=True)

# Split separately for each class
for class_index, class_name in enumerate(dataset.classes):

    class_images = [
        path
        for path, label in dataset.samples
        if label == class_index
    ]

    train_images, val_images = train_test_split(
        class_images,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE
    )

    train_class_dir = os.path.join(TRAIN_OUT, class_name)
    val_class_dir = os.path.join(VAL_OUT, class_name)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for src in train_images:
        shutil.copy2(
            src,
            os.path.join(train_class_dir, os.path.basename(src))
        )

    for src in val_images:
        shutil.copy2(
            src,
            os.path.join(val_class_dir, os.path.basename(src))
        )

    print(
        f"{class_name}: "
        f"train={len(train_images)}, "
        f"val={len(val_images)}"
    )

print("\nClean split created successfully.")

print("\nTrain:")
print(TRAIN_OUT)

print("\nValidation:")
print(VAL_OUT)