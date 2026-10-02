
from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

POSE_FILE = (
    ROOT / "dog_pose_outputs" / "analysis"
    / "pose_keypoints.csv"
)

RAW_CAM_FILE = (
    ROOT / "dog_pose_outputs" / "raw_gradcam"
    / "raw_gradcam_results.csv"
)

MAPPING_FILE = (
    ROOT / "dog_pose_outputs" / "analysis"
    / "exact_gradcam_pose_mapping.csv"
)

OUTPUT_DIR = (
    ROOT / "dog_pose_outputs" / "facial_attention_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

# Pose video dimensions
POSE_SIZE = 384

# Grad-CAM dimensions
CAM_SIZE = 224

# Grad-CAM preprocessing:
# Resize original image to 256x256, then center crop 224x224.
RESIZED_SIZE = 256
CROP_OFFSET = (RESIZED_SIZE - CAM_SIZE) / 2

# Circular neighborhood radius in Grad-CAM pixels.
REGION_RADIUS = 6

# Only use reliable facial landmarks.
KEYPOINTS = [
    "left_eye",
    "right_eye",
    "nose",
    "upper_jaw",
    "lower_jaw"
]

CONFIDENCE_THRESHOLD = 0.5

# --------------------------------------------------
# 3. LOAD AND MERGE DATA
# --------------------------------------------------

pose = pd.read_csv(POSE_FILE)
cam = pd.read_csv(RAW_CAM_FILE)
mapping = pd.read_csv(MAPPING_FILE)

# Verify that each frame occurs only once.
for name, df in [
    ("pose", pose),
    ("Grad-CAM", cam),
    ("mapping", mapping)
]:
    if df["frame"].duplicated().any():
        raise ValueError(f"Duplicate frame in {name} data.")

# Verify that the three datasets refer to the same frames.
data = mapping[
    ["frame", "original_filename", "pose_image"]
].merge(
    pose,
    on="frame",
    how="inner",
    validate="one_to_one"
).merge(
    cam[
        [
            "frame", "original_filename",
            "predicted_label", "true_label",
            "confidence", "raw_map_path"
        ]
    ],
    on="frame",
    how="inner",
    suffixes=("_mapping", "_cam"),
    validate="one_to_one"
)

if len(data) != len(mapping):
    raise ValueError("Not all frames were matched.")

if not (
    data["original_filename_mapping"]
    == data["original_filename_cam"]
).all():
    raise ValueError("Original filenames do not match.")

print("Matched frames:", len(data))

# --------------------------------------------------
# 4. COORDINATE TRANSFORMATION
# --------------------------------------------------

def pose_to_cam(x, y):
    """
    Transform coordinates from the 384x384 pose frame
    to the 224x224 Grad-CAM map.

    Assumes the pose video was created by resizing
    images directly to 384x384.
    """
    scale = RESIZED_SIZE / POSE_SIZE

    x_cam = x * scale - CROP_OFFSET
    y_cam = y * scale - CROP_OFFSET

    return x_cam, y_cam

# --------------------------------------------------
# 5. CREATE FACIAL REGIONS AND MEASURE ATTENTION
# --------------------------------------------------

yy, xx = np.mgrid[0:CAM_SIZE, 0:CAM_SIZE]

results = []

for _, row in data.iterrows():

    cam_path = Path(row["raw_map_path"])

    try:
        heatmap = np.load(cam_path).astype(np.float32)
    except Exception as e:
        print("Could not load:", cam_path, e)
        continue

    if heatmap.shape != (CAM_SIZE, CAM_SIZE):
        print("Unexpected heatmap shape:", cam_path)
        continue

    heatmap = np.nan_to_num(
        heatmap, nan=0.0, posinf=0.0, neginf=0.0
    )
    heatmap = np.clip(heatmap, 0, None)

    total_attention = float(heatmap.sum())

    facial_mask = np.zeros(
        (CAM_SIZE, CAM_SIZE), dtype=bool
    )

    used_points = []

    for name in KEYPOINTS:
        x = row.get(f"{name}_x", np.nan)
        y = row.get(f"{name}_y", np.nan)
        likelihood = row.get(
            f"{name}_likelihood", np.nan
        )

        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and np.isfinite(likelihood)
        ):
            continue

        if likelihood < CONFIDENCE_THRESHOLD:
            continue

        x_cam, y_cam = pose_to_cam(x, y)

        # Ignore points outside the cropped image.
        if not (
            0 <= x_cam < CAM_SIZE
            and 0 <= y_cam < CAM_SIZE
        ):
            continue

        # Circular neighborhood around each landmark.
        circle = (
            (xx - x_cam) ** 2
            + (yy - y_cam) ** 2
            <= REGION_RADIUS ** 2
        )

        facial_mask |= circle
        used_points.append(name)

    region_pixels = int(facial_mask.sum())
    valid_count = len(used_points)

    if valid_count == 0 or total_attention <= 1e-12:
        attention_fraction = np.nan
        area_fraction = np.nan
        enrichment = np.nan
    else:
        attention_in_region = float(
            heatmap[facial_mask].sum()
        )

        attention_fraction = (
            attention_in_region / total_attention
        )

        area_fraction = region_pixels / (
            CAM_SIZE * CAM_SIZE
        )

        # Compare observed attention to uniform
        # attention over the entire image.
        enrichment = (
            attention_fraction / area_fraction
            if area_fraction > 0 else np.nan
        )

    results.append({
        "frame": int(row["frame"]),
        "original_filename":
            row["original_filename_mapping"],
        "true_label": row["true_label"],
        "predicted_label": row["predicted_label"],
        "confidence": row["confidence"],
        "correct": (
            row["true_label"] == row["predicted_label"]
        ),
        "valid_facial_keypoints": valid_count,
        "used_keypoints": ",".join(used_points),
        "facial_region_pixels": region_pixels,
        "facial_area_fraction": area_fraction,
        "facial_attention_fraction": attention_fraction,
        "facial_attention_enrichment": enrichment
    })

# --------------------------------------------------
# 6. SAVE PER-IMAGE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

result_file = OUTPUT_DIR / "facial_attention_consistency.csv"
results_df.to_csv(result_file, index=False)

# --------------------------------------------------
# 7. SUMMARIZE RESULTS
# --------------------------------------------------

valid = results_df.dropna(
    subset=["facial_attention_fraction"]
).copy()

summary = (
    valid.groupby(
        ["true_label", "correct"], as_index=False
    )
    .agg(
        images=("frame", "count"),
        mean_facial_attention=(
            "facial_attention_fraction", "mean"
        ),
        median_facial_attention=(
            "facial_attention_fraction", "median"
        ),
        mean_area_fraction=(
            "facial_area_fraction", "mean"
        ),
        mean_enrichment=(
            "facial_attention_enrichment", "mean"
        ),
        mean_confidence=("confidence", "mean")
    )
)

summary_file = OUTPUT_DIR / "facial_attention_summary.csv"
summary.to_csv(summary_file, index=False)

# --------------------------------------------------
# 8. PRINT RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("FACIAL ATTENTION CONSISTENCY ANALYSIS")
print("=" * 65)

print("Images analyzed:", len(results_df))
print("Images with valid facial regions:", len(valid))
print(
    "Images without valid measurements:",
    len(results_df) - len(valid)
)

if len(valid):
    print(
        "\nOverall mean facial attention:",
        round(valid["facial_attention_fraction"].mean(), 4)
    )
    print(
        "Overall mean attention enrichment:",
        round(valid["facial_attention_enrichment"].mean(), 4)
    )

print("\nSummary by emotion and correctness:")
print(summary.to_string(index=False))

print("\nSaved per-image results:")
print(result_file)

print("\nSaved summary:")
print(summary_file)