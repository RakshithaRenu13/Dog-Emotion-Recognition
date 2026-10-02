
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

SOURCE = Path("Dataset")
TRAINVAL = Path("Dataset_trainval")
TEST = Path("Dataset_test")

CLASSES = ["angry", "curious", "happy", "sad", "sleepy"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEED = 42

# Prevent accidentally overwriting an existing split.
if TRAINVAL.exists() or TEST.exists():
    raise FileExistsError(
        "Dataset_trainval or Dataset_test already exists. "
        "Rename or remove the old split folders first."
    )

# Check the original dataset.
for emotion in CLASSES:
    folder = SOURCE / emotion
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing class folder: {folder}")

    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if len(images) < 3:
        raise ValueError(f"Not enough images in {folder}")

# Split each class separately to preserve class proportions.
for emotion in CLASSES:
    folder = SOURCE / emotion
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    trainval_images, test_images = train_test_split(
        images,
        test_size=0.15,
        random_state=SEED
    )

    # Create folders.
    (TRAINVAL / emotion).mkdir(parents=True, exist_ok=True)
    (TEST / emotion).mkdir(parents=True, exist_ok=True)

    # Copy images without modifying the original dataset.
    for image in trainval_images:
        shutil.copy2(image, TRAINVAL / emotion / image.name)

    for image in test_images:
        shutil.copy2(image, TEST / emotion / image.name)

    print(
        f"{emotion}: train+validation={len(trainval_images)}, "
        f"test={len(test_images)}"
    )

print("\nDataset split completed.")
print("Original Dataset folder remains unchanged.")