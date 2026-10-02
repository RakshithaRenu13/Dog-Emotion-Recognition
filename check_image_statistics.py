import os
import numpy as np
from PIL import Image

DATASETS = {
    "Train": "datasets/dog_emotions/train",
    "Validation": "datasets/dog_emotions/val",
    "Test": "Dataset_test"
}


def analyze_dataset(name, folder):

    brightness_values = []
    contrast_values = []
    aspect_ratios = []

    total = 0

    for root, _, files in os.walk(folder):

        for file in files:

            if not file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            ):
                continue

            path = os.path.join(root, file)

            try:
                with Image.open(path) as img:

                    img = img.convert("RGB")

                    arr = np.asarray(img).astype(np.float32)

                    # Convert RGB to approximate grayscale brightness
                    gray = (
                        0.299 * arr[:, :, 0]
                        + 0.587 * arr[:, :, 1]
                        + 0.114 * arr[:, :, 2]
                    )

                    brightness_values.append(gray.mean())
                    contrast_values.append(gray.std())

                    aspect_ratios.append(img.width / img.height)

                    total += 1

            except Exception as e:
                print("Error:", path, e)

    brightness_values = np.array(brightness_values)
    contrast_values = np.array(contrast_values)
    aspect_ratios = np.array(aspect_ratios)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Images:", total)

    print("\nBrightness")
    print("  Average:", round(brightness_values.mean(), 2))
    print("  Std:", round(brightness_values.std(), 2))
    print("  Min:", round(brightness_values.min(), 2))
    print("  Max:", round(brightness_values.max(), 2))

    print("\nContrast")
    print("  Average:", round(contrast_values.mean(), 2))
    print("  Std:", round(contrast_values.std(), 2))

    print("\nAspect Ratio")
    print("  Average:", round(aspect_ratios.mean(), 3))
    print("  Min:", round(aspect_ratios.min(), 3))
    print("  Max:", round(aspect_ratios.max(), 3))

    extreme = np.sum(
        (aspect_ratios < 0.75) |
        (aspect_ratios > 1.33)
    )

    print(
        "  Extreme aspect ratio (<0.75 or >1.33):",
        extreme,
        f"({100 * extreme / total:.2f}%)"
    )


for name, folder in DATASETS.items():

    if os.path.exists(folder):
        analyze_dataset(name, folder)
    else:
        print("\nMissing:", folder)