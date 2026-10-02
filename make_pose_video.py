from pathlib import Path
import cv2

PROJECT_ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

IMAGE_DIR = PROJECT_ROOT / "dog_pose_outputs" / "images"
OUTPUT_VIDEO = PROJECT_ROOT / "dog_pose_outputs" / "dog_test_images.mp4"

images = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

if not images:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

print(f"Found {len(images)} images")

first = cv2.imread(str(images[0]))

if first is None:
    raise RuntimeError(f"Could not read: {images[0]}")

height, width = first.shape[:2]

fps = 10

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (width, height)
)

for i, image_path in enumerate(images, 1):

    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"Skipping unreadable image: {image_path}")
        continue

    frame = cv2.resize(frame, (width, height))

    writer.write(frame)

    if i % 100 == 0:
        print(f"Written {i}/{len(images)}")

writer.release()

print("\nVideo created successfully:")
print(OUTPUT_VIDEO)
print(f"Frames: {len(images)}")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")