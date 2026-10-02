from torchvision.datasets import ImageFolder
from collections import Counter

datasets_to_check = {
    "Validation": "datasets/dog_emotions/val",
    "Test": "Dataset_test"
}

for name, path in datasets_to_check.items():

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    dataset = ImageFolder(path)

    counts = Counter(dataset.targets)

    print("Total images:", len(dataset))

    for index, class_name in enumerate(dataset.classes):
        print(
            f"{class_name}: "
            f"{counts[index]}"
        )