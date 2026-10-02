from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

POSE_FILE = (
    PROJECT_ROOT
    / "dog_pose_outputs"
    / "results"
    / "dog_test_images_superanimal_quadruped_hrnet_w32_fasterrcnn_resnet50_fpn_v2.h5"
)

OUTPUT_DIR = PROJECT_ROOT / "dog_pose_outputs" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "pose_keypoints.csv"
SUMMARY_CSV = OUTPUT_DIR / "pose_summary.csv"

# ============================================================
# LOAD DEEPLABCUT DATA
# ============================================================

print("=" * 70)
print("DOG POSE ANALYSIS")
print("=" * 70)

print(f"Loading: {POSE_FILE}")

df = pd.read_hdf(POSE_FILE)

print(f"Frames: {len(df)}")

# ============================================================
# GET BODY PARTS
# ============================================================

bodyparts = sorted(
    set(df.columns.get_level_values("bodyparts"))
)

print(f"Total keypoints: {len(bodyparts)}")

# ============================================================
# DOG-SPECIFIC KEYPOINTS
# Exclude antler points
# ============================================================

dog_keypoints = [
    bp for bp in bodyparts
    if "antler" not in bp
]

print(f"Dog-specific keypoints: {len(dog_keypoints)}")

print("\nKeypoints used:")
for bp in dog_keypoints:
    print("  ", bp)

# ============================================================
# CREATE FLAT DATAFRAME
# ============================================================

records = []

for frame_idx in range(len(df)):

    record = {
        "frame": frame_idx
    }

    for bp in dog_keypoints:

        x = df.iloc[frame_idx][
            (df.columns.get_level_values("bodyparts") == bp)
            & (df.columns.get_level_values("coords") == "x")
        ]

        y = df.iloc[frame_idx][
            (df.columns.get_level_values("bodyparts") == bp)
            & (df.columns.get_level_values("coords") == "y")
        ]

        likelihood = df.iloc[frame_idx][
            (df.columns.get_level_values("bodyparts") == bp)
            & (df.columns.get_level_values("coords") == "likelihood")
        ]

        record[f"{bp}_x"] = float(x.iloc[0])
        record[f"{bp}_y"] = float(y.iloc[0])
        record[f"{bp}_likelihood"] = float(likelihood.iloc[0])

    records.append(record)

flat_df = pd.DataFrame(records)

# ============================================================
# POSTURE FEATURES
# ============================================================

def distance(x1, y1, x2, y2):
    return np.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


# Head length
flat_df["head_length"] = distance(
    flat_df["nose_x"],
    flat_df["nose_y"],
    flat_df["neck_end_x"],
    flat_df["neck_end_y"]
)

# Body length
flat_df["body_length"] = distance(
    flat_df["neck_base_x"],
    flat_df["neck_base_y"],
    flat_df["back_end_x"],
    flat_df["back_end_y"]
)

# Neck length
flat_df["neck_length"] = distance(
    flat_df["neck_base_x"],
    flat_df["neck_base_y"],
    flat_df["neck_end_x"],
    flat_df["neck_end_y"]
)

# Tail length
flat_df["tail_length"] = distance(
    flat_df["tail_base_x"],
    flat_df["tail_base_y"],
    flat_df["tail_end_x"],
    flat_df["tail_end_y"]
)

# Front leg lengths
flat_df["front_left_leg"] = distance(
    flat_df["front_left_thai_x"],
    flat_df["front_left_thai_y"],
    flat_df["front_left_paw_x"],
    flat_df["front_left_paw_y"]
)

flat_df["front_right_leg"] = distance(
    flat_df["front_right_thai_x"],
    flat_df["front_right_thai_y"],
    flat_df["front_right_paw_x"],
    flat_df["front_right_paw_y"]
)

# Back leg lengths
flat_df["back_left_leg"] = distance(
    flat_df["back_left_thai_x"],
    flat_df["back_left_thai_y"],
    flat_df["back_left_paw_x"],
    flat_df["back_left_paw_y"]
)

flat_df["back_right_leg"] = distance(
    flat_df["back_right_thai_x"],
    flat_df["back_right_thai_y"],
    flat_df["back_right_paw_x"],
    flat_df["back_right_paw_y"]
)

# ============================================================
# CONFIDENCE STATISTICS
# ============================================================

likelihood_columns = [
    col for col in flat_df.columns
    if col.endswith("_likelihood")
]

flat_df["mean_pose_confidence"] = (
    flat_df[likelihood_columns].mean(axis=1)
)

flat_df["high_confidence_keypoints"] = (
    flat_df[likelihood_columns] >= 0.5
).sum(axis=1)

flat_df["low_confidence_keypoints"] = (
    flat_df[likelihood_columns] < 0.5
).sum(axis=1)

# ============================================================
# SAVE
# ============================================================

flat_df.to_csv(OUTPUT_CSV, index=False)

print("\nPose feature dataset saved:")
print(OUTPUT_CSV)

# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "Total frames",
        "Dog-specific keypoints",
        "Mean pose confidence",
        "Median pose confidence",
        "Mean high-confidence keypoints",
        "Mean low-confidence keypoints",
    ],
    "value": [
        len(flat_df),
        len(dog_keypoints),
        flat_df["mean_pose_confidence"].mean(),
        flat_df["mean_pose_confidence"].median(),
        flat_df["high_confidence_keypoints"].mean(),
        flat_df["low_confidence_keypoints"].mean(),
    ]
})

summary.to_csv(SUMMARY_CSV, index=False)

print("\nSummary:")
print(summary.to_string(index=False))

print("\nSummary saved:")
print(SUMMARY_CSV)

print("\n" + "=" * 70)
print("POSE ANALYSIS COMPLETED")
print("=" * 70)