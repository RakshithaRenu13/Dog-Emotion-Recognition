"""
Data loading utilities for dog emotion recognition.
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset

from ..config import EMOTION_CLASSES


def load_dataset(dataset_path: Path) -> Tuple[List[Path], List[str]]:
    """
    Load image paths and corresponding emotion labels.
    """

    image_paths = []
    labels = []

    for emotion in EMOTION_CLASSES:

        emotion_path = dataset_path / emotion

        if not emotion_path.exists():
            print(
                f"Warning: {emotion_path} not found, skipping..."
            )
            continue

        for img_file in sorted(emotion_path.iterdir()):

            if img_file.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png"
            ]:
                image_paths.append(img_file)
                labels.append(emotion)

    print(
        f"Found {len(image_paths)} images "
        f"across {len(set(labels))} emotion classes"
    )

    return image_paths, labels


class DogEmotionDataset(Dataset):
    """
    PyTorch dataset for dog emotion images.

    Important:
    This dataset does NOT shuffle the samples.

    Shuffling/splitting is handled externally so that
    training and validation datasets use exactly the
    same image indices.
    """

    def __init__(
        self,
        root_path: Path,
        transform=None
    ):

        self.samples = []
        self.transform = transform

        root_path = Path(root_path)

        for label_idx, emotion in enumerate(
            EMOTION_CLASSES
        ):

            emotion_path = root_path / emotion

            if not emotion_path.exists():
                continue

            for img_file in sorted(
                emotion_path.iterdir()
            ):

                if img_file.suffix.lower() in [
                    ".jpg",
                    ".jpeg",
                    ".png"
                ]:

                    self.samples.append(
                        (img_file, label_idx)
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple:

        img_path, label = self.samples[idx]

        try:

            image = Image.open(
                img_path
            ).convert("RGB")

        except Exception as e:

            print(
                f"Error loading {img_path}: {e}"
            )

            image = Image.new(
                "RGB",
                (224, 224),
                (0, 0, 0)
            )

        if self.transform:

            image = self.transform(image)

        return image, label, str(img_path)