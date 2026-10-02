import os
import shutil
import hashlib

SOURCE_TRAIN = "Dataset_trainval"
SOURCE_TEST = "Dataset_test"

CLEAN_TRAIN = "Dataset_trainval_clean"
CLEAN_TEST = "Dataset_test_clean"


def image_hash(path):
    h = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def get_images(folder):
    images = []

    for root, _, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                images.append(os.path.join(root, file))

    return images


print("=" * 70)
print("CREATING CLEAN DATASETS")
print("=" * 70)

# ---------------------------------------------------------
# Check source folders
# ---------------------------------------------------------

if not os.path.exists(SOURCE_TRAIN):
    raise FileNotFoundError(SOURCE_TRAIN)

if not os.path.exists(SOURCE_TEST):
    raise FileNotFoundError(SOURCE_TEST)

# ---------------------------------------------------------
# Remove old clean folders if they exist
# ---------------------------------------------------------

if os.path.exists(CLEAN_TRAIN):
    shutil.rmtree(CLEAN_TRAIN)

if os.path.exists(CLEAN_TEST):
    shutil.rmtree(CLEAN_TEST)

# ---------------------------------------------------------
# Create clean TrainVal copy
# ---------------------------------------------------------

print("\nCopying TrainVal dataset...")

shutil.copytree(SOURCE_TRAIN, CLEAN_TRAIN)

print("TrainVal copied.")

# ---------------------------------------------------------
# Build TrainVal hash database
# ---------------------------------------------------------

print("\nCalculating TrainVal image hashes...")

train_hashes = {}

train_images = get_images(SOURCE_TRAIN)

for path in train_images:

    h = image_hash(path)

    train_hashes.setdefault(h, []).append(path)

print("TrainVal images:", len(train_images))
print("Unique TrainVal hashes:", len(train_hashes))

# ---------------------------------------------------------
# Copy Test dataset while removing exact duplicates
# ---------------------------------------------------------

print("\nProcessing Test dataset...")

test_images = get_images(SOURCE_TEST)

removed = []
copied = 0

for source_path in test_images:

    h = image_hash(source_path)

    # Exact duplicate exists in TrainVal
    if h in train_hashes:

        removed.append(source_path)
        continue

    relative_path = os.path.relpath(
        source_path,
        SOURCE_TEST
    )

    destination_path = os.path.join(
        CLEAN_TEST,
        relative_path
    )

    os.makedirs(
        os.path.dirname(destination_path),
        exist_ok=True
    )

    shutil.copy2(
        source_path,
        destination_path
    )

    copied += 1


print("\n" + "=" * 70)
print("CLEANING COMPLETE")
print("=" * 70)

print("\nOriginal TrainVal:", len(train_images))
print("Clean TrainVal:", len(get_images(CLEAN_TRAIN)))

print("\nOriginal Test:", len(test_images))
print("Clean Test:", len(get_images(CLEAN_TEST)))

print("\nRemoved exact duplicate test images:", len(removed))

if removed:

    print("\nRemoved files:")

    for path in removed:
        print(" ", path)

print("\nCreated:")
print(" ", CLEAN_TRAIN)
print(" ", CLEAN_TEST)