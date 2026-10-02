import os
import hashlib

DATASETS = {
    "TrainVal": "Dataset_trainval",
    "Test": "Dataset_test",
}

def get_image_hash(path):
    hasher = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def collect_hashes(folder):
    hashes = {}

    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                path = os.path.join(root, file)

                try:
                    h = get_image_hash(path)
                    hashes.setdefault(h, []).append(path)
                except Exception as e:
                    print("Error:", path, e)

    return hashes


print("=" * 60)
print("CHECKING EXACT DUPLICATES")
print("=" * 60)

trainval_hashes = collect_hashes(DATASETS["TrainVal"])
test_hashes = collect_hashes(DATASETS["Test"])

print("\nTrainVal images:", sum(len(v) for v in trainval_hashes.values()))
print("TrainVal unique:", len(trainval_hashes))

print("\nTest images:", sum(len(v) for v in test_hashes.values()))
print("Test unique:", len(test_hashes))

common_hashes = set(trainval_hashes.keys()) & set(test_hashes.keys())

print("\nExact duplicate image files between TrainVal and Test:", len(common_hashes))

if common_hashes:
    print("\nWARNING: Duplicate images found!")

    shown = 0

    for h in common_hashes:
        print("\nTrainVal:")
        for path in trainval_hashes[h]:
            print(" ", path)

        print("Test:")
        for path in test_hashes[h]:
            print(" ", path)

        shown += 1

        if shown >= 10:
            print("\nShowing first 10 duplicate groups only.")
            break
else:
    print("\nNo exact duplicate images found.")