import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(r"C:\Users\acer\dogs-emotion-recognition")

GRADCAM_CSV = (
    PROJECT_DIR
    / "dog_pose_outputs"
    / "raw_gradcam"
    / "raw_gradcam_results.csv"
)

POSE_CSV = (
    PROJECT_DIR
    / "dog_pose_outputs"
    / "analysis"
    / "pose_keypoints.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "dog_pose_outputs"
    / "analysis"
    / "xai_bodypart"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# SETTINGS
# ============================================================

# Pose confidence threshold
POSE_THRESHOLD = 0.50

# Radius around each pose keypoint in the Grad-CAM map
KEYPOINT_RADIUS = 8

# Emotion classes
CLASS_NAMES = [
    "angry",
    "curious",
    "happy",
    "sad",
    "sleepy"
]

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("XAI BODY-PART ATTENTION ANALYSIS")
print("=" * 70)

print("\nLoading Grad-CAM CSV...")
gradcam_df = pd.read_csv(GRADCAM_CSV)

print("Grad-CAM rows:", len(gradcam_df))

print("\nLoading pose CSV...")
pose_df = pd.read_csv(POSE_CSV)

print("Pose rows:", len(pose_df))

print("\nGrad-CAM columns:")
print(gradcam_df.columns.tolist())

print("\nPose columns:")
print(pose_df.columns.tolist())


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Find the first matching column from candidate names.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def get_keypoint_columns(df, keypoint):
    """
    Find x, y, confidence columns for a keypoint.

    Supports common naming styles:
        nose_x
        nose_y
        nose_likelihood

    Also supports:
        nose.x
        nose.y
        nose.confidence
    """

    columns_lower = {str(c).lower(): c for c in df.columns}

    x_candidates = [
        f"{keypoint}_x",
        f"{keypoint}.x",
        f"{keypoint}x",
        f"x_{keypoint}",
    ]

    y_candidates = [
        f"{keypoint}_y",
        f"{keypoint}.y",
        f"{keypoint}y",
        f"y_{keypoint}",
    ]

    conf_candidates = [
        f"{keypoint}_likelihood",
        f"{keypoint}_confidence",
        f"{keypoint}_conf",
        f"{keypoint}.likelihood",
        f"{keypoint}.confidence",
        f"{keypoint}_score",
    ]

    x_col = None
    y_col = None
    conf_col = None

    for c in x_candidates:
        if c.lower() in columns_lower:
            x_col = columns_lower[c.lower()]
            break

    for c in y_candidates:
        if c.lower() in columns_lower:
            y_col = columns_lower[c.lower()]
            break

    for c in conf_candidates:
        if c.lower() in columns_lower:
            conf_col = columns_lower[c.lower()]
            break

    return x_col, y_col, conf_col


def load_gradcam_map(path):
    """
    Load Grad-CAM numpy map and normalize to [0, 1].
    """

    if not os.path.exists(path):
        return None

    cam = np.load(path)

    cam = np.asarray(cam, dtype=np.float32)

    if cam.ndim != 2:
        return None

    cam_min = np.nanmin(cam)
    cam_max = np.nanmax(cam)

    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    return cam


def get_attention_around_point(cam, x, y, radius=8):
    """
    Calculate mean Grad-CAM activation around a pose keypoint.
    """

    h, w = cam.shape

    x = int(round(x))
    y = int(round(y))

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)

    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    patch = cam[y1:y2, x1:x2]

    if patch.size == 0:
        return np.nan

    return float(np.mean(patch))


# ============================================================
# KEYPOINT DEFINITIONS
# ============================================================

KEYPOINTS = [
    "nose",

    "left_eye",
    "right_eye",

    "left_earbase",
    "left_earend",
    "right_earbase",
    "right_earend",

    "upper_jaw",
    "lower_jaw",
    "mouth_end_left",
    "mouth_end_right",

    "neck_base",
    "neck_end",

    "body_middle_left",
    "body_middle_right",
    "back_middle",

    "front_left_thai",
    "front_left_knee",
    "front_left_paw",

    "front_right_thai",
    "front_right_knee",
    "front_right_paw",

    "back_left_thai",
    "back_left_knee",
    "back_left_paw",

    "back_right_thai",
    "back_right_knee",
    "back_right_paw",

    "tail_base",
    "tail_end",

    "belly_bottom",
    "throat_base",
    "throat_end",
]


# ============================================================
# BODY REGION DEFINITIONS
# ============================================================

REGIONS = {

    "head": [
        "nose",
        "left_eye",
        "right_eye",
        "neck_base",
        "neck_end",
        "upper_jaw",
        "lower_jaw",
    ],

    "ears": [
        "left_earbase",
        "left_earend",
        "right_earbase",
        "right_earend",
    ],

    "mouth": [
        "upper_jaw",
        "lower_jaw",
        "mouth_end_left",
        "mouth_end_right",
    ],

    "body": [
        "body_middle_left",
        "body_middle_right",
        "back_middle",
        "belly_bottom",
        "throat_base",
        "throat_end",
    ],

    "front_legs": [
        "front_left_thai",
        "front_left_knee",
        "front_left_paw",
        "front_right_thai",
        "front_right_knee",
        "front_right_paw",
    ],

    "hind_legs": [
        "back_left_thai",
        "back_left_knee",
        "back_left_paw",
        "back_right_thai",
        "back_right_knee",
        "back_right_paw",
    ],

    "tail": [
        "tail_base",
        "tail_end",
    ],
}


# ============================================================
# FIND FRAME COLUMNS
# ============================================================

gradcam_frame_col = find_column(
    gradcam_df,
    ["frame", "frame_id", "image_id"]
)

pose_frame_col = find_column(
    pose_df,
    ["frame", "frame_id", "image_id"]
)

if gradcam_frame_col is None:
    raise ValueError(
        "Could not find frame column in Grad-CAM CSV."
    )

if pose_frame_col is None:
    raise ValueError(
        "Could not find frame column in pose CSV."
    )


# ============================================================
# CREATE POSE LOOKUP
# ============================================================

pose_lookup = {}

for _, row in pose_df.iterrows():

    frame = row[pose_frame_col]

    try:
        frame = int(frame)
    except:
        continue

    pose_lookup[frame] = row


# ============================================================
# PROCESS IMAGES
# ============================================================

results = []

print("\nStarting XAI analysis...\n")

for idx, row in gradcam_df.iterrows():

    frame = row[gradcam_frame_col]

    try:
        frame = int(frame)
    except:
        continue

    # --------------------------------------------------------
    # Load Grad-CAM
    # --------------------------------------------------------

    map_path = row.get("raw_map_path", None)

    if pd.isna(map_path) or map_path is None:
        continue

    cam = load_gradcam_map(str(map_path))

    if cam is None:
        continue

    cam_h, cam_w = cam.shape

    # --------------------------------------------------------
    # Get corresponding pose
    # --------------------------------------------------------

    if frame not in pose_lookup:
        continue

    pose_row = pose_lookup[frame]

    # --------------------------------------------------------
    # Collect keypoints
    # --------------------------------------------------------

    keypoints = {}

    for kp in KEYPOINTS:

        x_col, y_col, conf_col = get_keypoint_columns(
            pose_df,
            kp
        )

        if x_col is None or y_col is None:
            continue

        try:
            x = float(pose_row[x_col])
            y = float(pose_row[y_col])
        except:
            continue

        # Confidence
        if conf_col is not None:
            try:
                conf = float(pose_row[conf_col])
            except:
                conf = 0.0
        else:
            conf = 1.0

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        if conf < POSE_THRESHOLD:
            continue

        keypoints[kp] = {
            "x": x,
            "y": y,
            "confidence": conf
        }

    # --------------------------------------------------------
    # Determine coordinate scale
    #
    # Pose coordinates are normally in original-image
    # coordinates, while Grad-CAM is 224x224.
    #
    # We use the actual image dimensions when available.
    # --------------------------------------------------------

    image_filename = row.get(
        "original_filename",
        None
    )

    scale_x = 1.0
    scale_y = 1.0

    if image_filename is not None and not pd.isna(image_filename):

        image_path = (
            PROJECT_DIR
            / "Dataset_test_clean"
            / str(image_filename)
        )

        if image_path.exists():

            try:
                with Image.open(image_path) as im:

                    original_w, original_h = im.size

                scale_x = cam_w / original_w
                scale_y = cam_h / original_h

            except Exception:
                pass

    # --------------------------------------------------------
    # Calculate attention for every keypoint
    # --------------------------------------------------------

    kp_attention = {}

    for kp, point in keypoints.items():

        cam_x = point["x"] * scale_x
        cam_y = point["y"] * scale_y

        attention = get_attention_around_point(
            cam,
            cam_x,
            cam_y,
            radius=KEYPOINT_RADIUS
        )

        kp_attention[kp] = attention

    # --------------------------------------------------------
    # Calculate body-region attention
    # --------------------------------------------------------

    region_values = {}

    for region_name, region_keypoints in REGIONS.items():

        values = []

        for kp in region_keypoints:

            if kp in kp_attention:

                value = kp_attention[kp]

                if np.isfinite(value):
                    values.append(value)

        if len(values) > 0:
            region_values[region_name] = float(
                np.mean(values)
            )
        else:
            region_values[region_name] = np.nan

    # --------------------------------------------------------
    # Convert regional attention to percentages
    # --------------------------------------------------------

    valid_values = [
        v for v in region_values.values()
        if np.isfinite(v)
    ]

    total_attention = sum(valid_values)

    region_percentages = {}

    for region_name, value in region_values.items():

        if np.isfinite(value) and total_attention > 0:

            region_percentages[region_name] = (
                100.0 * value / total_attention
            )

        else:

            region_percentages[region_name] = np.nan

    # --------------------------------------------------------
    # Find most important body region
    # --------------------------------------------------------

    valid_percentages = {
        k: v
        for k, v in region_percentages.items()
        if np.isfinite(v)
    }

    if valid_percentages:

        dominant_region = max(
            valid_percentages,
            key=valid_percentages.get
        )

        dominant_value = valid_percentages[
            dominant_region
        ]

    else:

        dominant_region = "unknown"
        dominant_value = np.nan

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    result = {
        "frame": frame,

        "original_filename":
            row.get("original_filename", ""),

        "true_label":
            row.get("true_label", ""),

        "predicted_label":
            row.get("predicted_label", ""),

        "confidence":
            row.get("confidence", np.nan),

        "num_reliable_keypoints":
            len(keypoints),

        "head_attention":
            region_values["head"],

        "ears_attention":
            region_values["ears"],

        "mouth_attention":
            region_values["mouth"],

        "body_attention":
            region_values["body"],

        "front_legs_attention":
            region_values["front_legs"],

        "hind_legs_attention":
            region_values["hind_legs"],

        "tail_attention":
            region_values["tail"],

        "head_percent":
            region_percentages["head"],

        "ears_percent":
            region_percentages["ears"],

        "mouth_percent":
            region_percentages["mouth"],

        "body_percent":
            region_percentages["body"],

        "front_legs_percent":
            region_percentages["front_legs"],

        "hind_legs_percent":
            region_percentages["hind_legs"],

        "tail_percent":
            region_percentages["tail"],

        "dominant_body_region":
            dominant_region,

        "dominant_body_region_percent":
            dominant_value,
    }

    results.append(result)

    if (idx + 1) % 100 == 0:
        print(
            f"Processed {idx + 1}/{len(gradcam_df)}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

output_csv = (
    OUTPUT_DIR
    / "bodypart_xai_results.csv"
)

result_df.to_csv(
    output_csv,
    index=False
)

print("\n" + "=" * 70)
print("XAI ANALYSIS COMPLETE")
print("=" * 70)

print("Images analyzed:", len(result_df))

print("\nOutput:")
print(output_csv)

# ============================================================
# SUMMARY
# ============================================================

if len(result_df) > 0:

    print("\nAverage body-region attention:")

    summary_columns = [
        "head_percent",
        "ears_percent",
        "mouth_percent",
        "body_percent",
        "front_legs_percent",
        "hind_legs_percent",
        "tail_percent",
    ]

    print(
        result_df[summary_columns]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nDominant body regions:")

    print(
        result_df["dominant_body_region"]
        .value_counts()
    )

    print("\nAverage attention by emotion:")

    emotion_summary = (
        result_df
        .groupby("predicted_label")[
            summary_columns
        ]
        .mean()
    )

    print(emotion_summary)

    emotion_output = (
        OUTPUT_DIR
        / "bodypart_xai_by_emotion.csv"
    )

    emotion_summary.to_csv(
        emotion_output
    )

    print(
        "\nEmotion summary saved to:"
    )

    print(emotion_output)

print("\nDONE.")