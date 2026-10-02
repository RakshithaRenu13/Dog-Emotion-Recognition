from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

# Original test dataset
TEST_DIR = PROJECT_ROOT / "Dataset_test_clean"

# Numbered images used for pose estimation
POSE_IMAGE_DIR = PROJECT_ROOT / "dog_pose_outputs" / "images"

# Existing Grad-CAM results
GRADCAM_FILE = (
    PROJECT_ROOT
    / "gradcam_outputs"
    / "gradcam_results.csv"
)

# Output
OUTPUT_FILE = (
    PROJECT_ROOT
    / "dog_pose_outputs"
    / "analysis"
    / "exact_gradcam_pose_mapping.csv"
)


# ---------------------------------------------------------
# 1. Get original test images in sorted order
# ---------------------------------------------------------
original_images = sorted(
    list(TEST_DIR.rglob("*.jpg")) +
    list(TEST_DIR.rglob("*.jpeg")) +
    list(TEST_DIR.rglob("*.png"))
)

# ---------------------------------------------------------
# 2. Get numbered pose images
# ---------------------------------------------------------
pose_images = sorted(
    list(POSE_IMAGE_DIR.glob("*.jpg")) +
    list(POSE_IMAGE_DIR.glob("*.jpeg")) +
    list(POSE_IMAGE_DIR.glob("*.png"))
)

print("Original test images:", len(original_images))
print("Pose images:", len(pose_images))

if len(original_images) != len(pose_images):
    raise ValueError(
        f"Image count mismatch: "
        f"{len(original_images)} original vs "
        f"{len(pose_images)} pose"
    )


# ---------------------------------------------------------
# 3. Create exact image correspondence
# ---------------------------------------------------------
mapping = pd.DataFrame({
    "frame": range(len(pose_images)),
    "pose_image": [x.name for x in pose_images],
    "original_filename": [x.name for x in original_images],
    "original_path": [str(x) for x in original_images],
    "true_label_from_path": [
        x.parent.name for x in original_images
    ]
})


# ---------------------------------------------------------
# 4. Load Grad-CAM results
# ---------------------------------------------------------
gradcam = pd.read_csv(GRADCAM_FILE)

gradcam["original_filename"] = gradcam["image"].apply(
    lambda x: Path(x).name
)


# ---------------------------------------------------------
# 5. Merge Grad-CAM results
# ---------------------------------------------------------
merged = mapping.merge(
    gradcam[
        [
            "original_filename",
            "true_label",
            "predicted_label",
            "confidence",
            "predicted_class_index"
        ]
    ],
    on="original_filename",
    how="left"
)


# ---------------------------------------------------------
# 6. Check matching
# ---------------------------------------------------------
missing = merged["predicted_label"].isna().sum()

print("\nMissing Grad-CAM matches:", missing)

if missing > 0:
    print("\nMissing filenames:")
    print(
        merged[
            merged["predicted_label"].isna()
        ]["original_filename"].head(20).to_string(index=False)
    )


# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------
merged.to_csv(OUTPUT_FILE, index=False)

print("\nExact mapping saved:")
print(OUTPUT_FILE)

print("\nFirst 10 rows:")
print(
    merged[
        [
            "frame",
            "pose_image",
            "original_filename",
            "true_label",
            "predicted_label",
            "confidence"
        ]
    ]
    .head(10)
    .to_string(index=False)
)