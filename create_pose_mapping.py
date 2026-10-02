from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

IMAGE_DIR = PROJECT_ROOT / "dog_pose_outputs" / "images"
POSE_FILE = (
    PROJECT_ROOT
    / "dog_pose_outputs"
    / "analysis"
    / "pose_keypoints.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dog_pose_outputs"
    / "analysis"
    / "pose_image_mapping.csv"
)

# Get images in exactly the same order used to create the video
images = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

pose_df = pd.read_csv(POSE_FILE)

print("Images:", len(images))
print("Pose frames:", len(pose_df))

if len(images) != len(pose_df):
    raise ValueError(
        f"Mismatch: {len(images)} images vs "
        f"{len(pose_df)} pose frames"
    )

mapping = pd.DataFrame({
    "frame": range(len(images)),
    "image_filename": [x.name for x in images],
    "image_path": [str(x) for x in images]
})

# Add pose confidence
likelihood_cols = [
    c for c in pose_df.columns
    if c.endswith("_likelihood")
]

mapping["mean_pose_confidence"] = (
    pose_df[likelihood_cols].mean(axis=1)
)

mapping["high_confidence_keypoints"] = (
    pose_df[likelihood_cols] >= 0.5
).sum(axis=1)

mapping.to_csv(OUTPUT_FILE, index=False)

print("\nMapping saved:")
print(OUTPUT_FILE)

print("\nFirst 10 mappings:")
print(mapping.head(10).to_string(index=False))