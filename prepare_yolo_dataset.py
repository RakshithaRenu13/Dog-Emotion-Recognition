
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

SOURCE = Path("Dataset_segmented_trainval")
DEST = Path("datasets/dog_emotions")
VAL_SIZE = 0.20
SEED = 42

if not SOURCE.exists():
    raise FileNotFoundError(f"Dataset not found: {SOURCE}")

if DEST.exists():
    raise FileExistsError(
        f"{DEST} already exists. Rename or remove it before preparing again."
    )

classes = ["angry", "curious", "happy", "sad", "sleepy"]

paths = []
labels = []

for label, class_name in enumerate(classes):
    folder = SOURCE / class_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing class folder: {folder}")

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in
        [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    )

    if not images:
        raise ValueError(f"No images found in {folder}")

    paths.extend(images)
    labels.extend([label] * len(images))

train_paths, val_paths, train_labels, val_labels = train_test_split(
    paths,
    labels,
    test_size=VAL_SIZE,
    random_state=SEED,
    stratify=labels
)

for split, split_paths, split_labels in [
    ("train", train_paths, train_labels),
    ("val", val_paths, val_labels)
]:
    for path, label in zip(split_paths, split_labels):
        class_name = classes[label]
        target_dir = DEST / split / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target_dir / path.name)

print("Dataset preparation complete.")
print("Training images:", len(train_paths))
print("Validation images:", len(val_paths))
print("Classes:", classes)
print("Dataset location:", DEST)