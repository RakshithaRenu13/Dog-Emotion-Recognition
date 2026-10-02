
from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["angry", "curious", "happy", "sad", "sleepy"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def segment_image(model, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    h, w = image.shape[:2]
    results = model.predict(image, verbose=False)
    result = results[0]

    if result.boxes is None or result.masks is None:
        return None

    # Find the largest detected dog.
    dog_indices = [
        i for i, cls in enumerate(result.boxes.cls.tolist())
        if int(cls) == 16  # COCO dog class
    ]

    if not dog_indices:
        return None

    best = max(
        dog_indices,
        key=lambda i: float(
            (result.boxes.xyxy[i][2] - result.boxes.xyxy[i][0]) *
            (result.boxes.xyxy[i][3] - result.boxes.xyxy[i][1])
        )
    )

    # Convert the dog's segmentation polygon into a mask.
    mask = np.zeros((h, w), dtype=np.uint8)
    polygon = result.masks.xy[best]

    if len(polygon) < 3:
        return None

    polygon = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    # Use the detected box with padding to retain the whole dog.
    x1, y1, x2, y2 = result.boxes.xyxy[best].cpu().numpy()
    pad_x = int((x2 - x1) * 0.12)
    pad_y = int((y2 - y1) * 0.12)

    x1 = max(0, int(x1) - pad_x)
    y1 = max(0, int(y1) - pad_y)
    x2 = min(w, int(x2) + pad_x)
    y2 = min(h, int(y2) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # Replace the background with neutral gray.
    crop[crop_mask == 0] = 127

    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        raise FileNotFoundError(input_dir)

    model = YOLO("yolov8n-seg.pt")
    processed = 0
    fallback = 0

    for emotion in CLASSES:
        source_folder = input_dir / emotion
        target_folder = output_dir / emotion
        target_folder.mkdir(parents=True, exist_ok=True)

        if not source_folder.is_dir():
            print(f"Missing class folder: {source_folder}")
            continue

        for image_path in source_folder.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            segmented = segment_image(model, image_path)

            if segmented is None:
                # Keep the original image if no dog is detected.
                original = cv2.imread(str(image_path))
                if original is None:
                    print("Unreadable:", image_path)
                    continue
                segmented = original
                fallback += 1

            output_path = target_folder / (image_path.stem + ".jpg")
            cv2.imwrite(str(output_path), segmented)
            processed += 1

        print(f"Completed: {emotion}")

    print("Total saved:", processed)
    print("Images without detected dog:", fallback)
    print("Output:", output_dir.resolve())


if __name__ == "__main__":
    main()