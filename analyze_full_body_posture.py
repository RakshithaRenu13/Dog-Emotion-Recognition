
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")
INPUT = ROOT / "dog_pose_outputs" / "analysis" / "pose_keypoints.csv"
OUT = ROOT / "dog_pose_outputs" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Calculate angle of a line relative to horizontal.
def angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2-y1, x2-x1))

# Body alignment angle
df["body_angle"] = angle(
    df["neck_base_x"], df["neck_base_y"],
    df["back_end_x"], df["back_end_y"]
)

# Head orientation angle
df["head_angle"] = angle(
    df["neck_end_x"], df["neck_end_y"],
    df["nose_x"], df["nose_y"]
)

# Difference between front and rear body height.
df["body_height_difference"] = (
    df["neck_base_y"] - df["back_end_y"]
)

# Calculate posture only when keypoint confidence is adequate.
def reliable(row, points, threshold=0.5):
    return all(
        pd.notna(row.get(p + "_likelihood")) and
        row[p + "_likelihood"] >= threshold
        for p in points
    )

body_points = ["neck_base", "back_end"]
head_points = ["neck_end", "nose"]

df["body_angle_reliable"] = df.apply(
    lambda r: reliable(r, body_points), axis=1
)
df["head_angle_reliable"] = df.apply(
    lambda r: reliable(r, head_points), axis=1
)

df.loc[~df["body_angle_reliable"], "body_angle"] = np.nan
df.loc[~df["head_angle_reliable"], "head_angle"] = np.nan

# Save frame-level posture features.
output = OUT / "full_body_posture.csv"
df.to_csv(output, index=False)

summary = pd.DataFrame({
    "metric": [
        "Total frames",
        "Reliable body angle frames",
        "Reliable head angle frames",
        "Mean body angle",
        "Mean head angle"
    ],
    "value": [
        len(df),
        int(df["body_angle_reliable"].sum()),
        int(df["head_angle_reliable"].sum()),
        df["body_angle"].mean(),
        df["head_angle"].mean()
    ]
})

summary.to_csv(OUT / "posture_summary.csv", index=False)

print("=" * 60)
print("FULL-BODY POSTURE ANALYSIS COMPLETED")
print("=" * 60)
print(summary.to_string(index=False))
print("\nSaved:", output)
print("Saved:", OUT / "posture_summary.csv")