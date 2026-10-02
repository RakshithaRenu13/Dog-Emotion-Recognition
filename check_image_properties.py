import os
from PIL import Image
from collections import Counter

DATASETS = {
    "Train": "datasets/dog_emotions/train",
    "Validation": "datasets/dog_emotions/val",
    "Test": "Dataset_test"
}


def analyze_dataset(name, folder):

    widths = []
    heights = []
    modes = []
    formats = []

    total = 0
    failed = 0

    for root, _, files in os.walk(folder):

        for file in files:

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                continue

            path = os.path.join(root, file)

            try:
                with Image.open(path) as img:

                    widths.append(img.width)
                    heights.append(img.height)
                    modes.append(img.mode)
                    formats.append(img.format)

                    total += 1

            except Exception:
                failed += 1

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Images:", total)

    if total == 0:
        return

    print("Failed:", failed)

    print("\nWidth:")
    print("  Min:", min(widths))
    print("  Max:", max(widths))
    print("  Average:", round(sum(widths) / len(widths), 2))

    print("\nHeight:")
    print("  Min:", min(heights))
    print("  Max:", max(heights))
    print("  Average:", round(sum(heights) / len(heights), 2))

    print("\nModes:")
    print(dict(Counter(modes)))

    print("\nFormats:")
    print(dict(Counter(formats)))


for name, folder in DATASETS.items():

    if os.path.exists(folder):
        analyze_dataset(name, folder)
    else:
        print("\nMissing:", folder)