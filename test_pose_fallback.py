from pathlib import Path
import pandas as pd
import numpy as np
import cv2

# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

FAILURE_CSV = (
    ROOT
    / "dog_pose_outputs"
    / "analysis"
    / "dog_detection_failures.csv"
)

POSE_CSV = (
    ROOT
    / "dog_pose_outputs"
    / "analysis"
    / "pose_keypoints.csv"
)

GRADCAM_CSV = (
    ROOT
    / "dog_pose_outputs"
    / "raw_gradcam"
    / "raw_gradcam_results.csv"
)

OUTPUT_CSV = (
    ROOT
    / "dog_pose_outputs"
    / "analysis"
    / "pose_fallback_experiment.csv"
)


# ============================================================
# PARAMETERS
# ============================================================

LIKELIHOOD_THRESHOLD = 0.50

# Expand pose bounding box by 20%
BOX_EXPANSION = 0.20

# Minimum number of reliable keypoints
MIN_KEYPOINTS = 5


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("POSE FALLBACK EXPERIMENT")
print("=" * 75)

failures = pd.read_csv(FAILURE_CSV)
pose = pd.read_csv(POSE_CSV)
gradcam = pd.read_csv(GRADCAM_CSV)

print("\nFailure rows:", len(failures))
print("Pose rows:", len(pose))
print("Grad-CAM rows:", len(gradcam))


# ============================================================
# MERGE
# ============================================================

df = failures.merge(
    pose,
    on="frame",
    how="left",
    suffixes=("", "_pose")
)

df = df.merge(
    gradcam[
        [
            "frame",
            "raw_map_path",
            "map_height",
            "map_width",
            "map_max",
            "map_mean"
        ]
    ],
    on="frame",
    how="left"
)

print("\nMerged rows:", len(df))


# ============================================================
# IDENTIFY POSE KEYPOINTS
# ============================================================

keypoint_names = []

for col in pose.columns:

    if col.endswith("_x"):

        name = col[:-2]

        if (
            f"{name}_y" in pose.columns
            and f"{name}_likelihood" in pose.columns
        ):
            keypoint_names.append(name)


print("\nPose keypoints detected:", len(keypoint_names))

print(
    keypoint_names
)


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

DATASET = ROOT / "Dataset_test_clean"

image_files = sorted(
    [
        p
        for p in DATASET.rglob("*")
        if p.is_file()
        and p.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png"
        ]
    ]
)

print("\nDataset images:", len(image_files))


def get_image_path(frame):

    try:
        index = int(frame)
    except:
        return None

    if 0 <= index < len(image_files):

        return image_files[index]

    return None


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def load_gradcam(path):

    if pd.isna(path):
        return None

    path = Path(str(path))

    if not path.exists():
        return None

    try:

        cam = np.load(path)

    except Exception:
        return None

    if cam is None:
        return None

    cam = np.asarray(cam).astype(np.float32)

    # Remove negative activation
    cam = np.maximum(cam, 0)

    # Normalize
    max_value = cam.max()

    if max_value > 0:

        cam = cam / max_value

    return cam


# ============================================================
# SCALE POSE COORDINATES TO IMAGE
# ============================================================

def get_pose_points(row, image_width, image_height):

    points = []

    for name in keypoint_names:

        x_col = f"{name}_x"
        y_col = f"{name}_y"
        l_col = f"{name}_likelihood"

        try:

            x = float(row[x_col])
            y = float(row[y_col])
            likelihood = float(row[l_col])

        except:

            continue

        if not np.isfinite(x):
            continue

        if not np.isfinite(y):
            continue

        if not np.isfinite(likelihood):
            continue

        if likelihood < LIKELIHOOD_THRESHOLD:
            continue

        points.append(
            (x, y, likelihood)
        )

    if len(points) == 0:

        return []

    # --------------------------------------------------------
    # Determine coordinate system
    #
    # Some pose systems store coordinates in original pixels.
    # Others store normalized [0,1] coordinates.
    # --------------------------------------------------------

    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)

    # Normalized coordinates
    if max_x <= 1.5 and max_y <= 1.5:

        points = [
            (
                x * image_width,
                y * image_height,
                likelihood
            )
            for x, y, likelihood in points
        ]

    # 224x224 coordinate system
    elif max_x <= 250 and max_y <= 250:

        # If image is approximately 224-based,
        # convert to original image coordinates.
        points = [
            (
                x * image_width / 224.0,
                y * image_height / 224.0,
                likelihood
            )
            for x, y, likelihood in points
        ]

    return points


# ============================================================
# CREATE POSE BOX
# ============================================================

def create_pose_box(points, image_width, image_height):

    if len(points) < MIN_KEYPOINTS:

        return None

    xs = np.array(
        [p[0] for p in points],
        dtype=np.float32
    )

    ys = np.array(
        [p[1] for p in points],
        dtype=np.float32
    )

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())

    width = x2 - x1
    height = y2 - y1

    # Avoid zero-sized box
    if width < 5 or height < 5:

        return None

    # Expand
    x1 -= width * BOX_EXPANSION
    x2 += width * BOX_EXPANSION

    y1 -= height * BOX_EXPANSION
    y2 += height * BOX_EXPANSION

    # Clamp
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(image_width - 1, x2)
    y2 = min(image_height - 1, y2)

    return (
        int(x1),
        int(y1),
        int(x2),
        int(y2)
    )


# ============================================================
# GRAD-CAM INSIDE BOX
# ============================================================

def gradcam_inside_box(
    cam,
    box,
    image_width,
    image_height
):

    if cam is None:
        return np.nan

    if box is None:
        return np.nan

    x1, y1, x2, y2 = box

    cam_h, cam_w = cam.shape[:2]

    # Convert original image coordinates
    # into Grad-CAM coordinates.

    gx1 = int(
        x1 / image_width * cam_w
    )

    gx2 = int(
        x2 / image_width * cam_w
    )

    gy1 = int(
        y1 / image_height * cam_h
    )

    gy2 = int(
        y2 / image_height * cam_h
    )

    gx1 = max(0, min(cam_w - 1, gx1))
    gx2 = max(0, min(cam_w, gx2))

    gy1 = max(0, min(cam_h - 1, gy1))
    gy2 = max(0, min(cam_h, gy2))

    if gx2 <= gx1 or gy2 <= gy1:

        return np.nan

    box_activation = cam[
        gy1:gy2,
        gx1:gx2
    ].sum()

    total_activation = cam.sum()

    if total_activation <= 0:

        return np.nan

    return float(
        box_activation /
        total_activation
    )


# ============================================================
# POSE INSIDE BOX
# ============================================================

def pose_inside_box(
    points,
    box
):

    if box is None:
        return np.nan

    if len(points) == 0:
        return np.nan

    x1, y1, x2, y2 = box

    inside = 0

    for x, y, likelihood in points:

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):

            inside += 1

    return float(
        inside / len(points)
    )


# ============================================================
# MAIN LOOP
# ============================================================

results = []

print("\nProcessing failures...")

for count, (_, row) in enumerate(
    df.iterrows(),
    start=1
):

    frame = row["frame"]

    image_path = get_image_path(frame)

    if image_path is None:

        results.append({
            "frame": frame,
            "fallback_status": "image_not_found"
        })

        continue

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        results.append({
            "frame": frame,
            "fallback_status": "image_read_failed"
        })

        continue

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # Pose points
    # --------------------------------------------------------

    points = get_pose_points(
        row,
        image_width,
        image_height
    )

    reliable_count = len(points)

    # --------------------------------------------------------
    # Pose box
    # --------------------------------------------------------

    box = create_pose_box(
        points,
        image_width,
        image_height
    )

    if box is None:

        results.append({
            "frame": frame,
            "filename": row["filename"],
            "failure_reason": row["failure_reason"],
            "fallback_status": "insufficient_pose",
            "reliable_pose_keypoints": reliable_count,
            "pose_box_x1": np.nan,
            "pose_box_y1": np.nan,
            "pose_box_x2": np.nan,
            "pose_box_y2": np.nan,
            "pose_box_area_fraction": np.nan,
            "gradcam_inside_pose_box": np.nan,
            "pose_inside_box_fraction": np.nan
        })

        continue

    x1, y1, x2, y2 = box

    box_area = (
        (x2 - x1) *
        (y2 - y1)
    )

    image_area = (
        image_width *
        image_height
    )

    box_fraction = (
        box_area /
        image_area
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    cam = load_gradcam(
        row["raw_map_path"]
    )

    cam_fraction = gradcam_inside_box(
        cam,
        box,
        image_width,
        image_height
    )

    # --------------------------------------------------------
    # Pose coverage
    # --------------------------------------------------------

    pose_fraction = pose_inside_box(
        points,
        box
    )

    results.append({

        "frame": frame,

        "filename":
            row["filename"],

        "failure_reason":
            row["failure_reason"],

        "fallback_status":
            "pose_box_success",

        "reliable_pose_keypoints":
            reliable_count,

        "pose_box_x1":
            x1,

        "pose_box_y1":
            y1,

        "pose_box_x2":
            x2,

        "pose_box_y2":
            y2,

        "pose_box_area_fraction":
            box_fraction,

        "gradcam_inside_pose_box":
            cam_fraction,

        "pose_inside_box_fraction":
            pose_fraction
    })

    if count % 50 == 0:

        print(
            f"Processed {count}/{len(df)}"
        )


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(results)

result.to_csv(
    OUTPUT_CSV,
    index=False
)

print("\n" + "=" * 75)
print("POSE FALLBACK RESULTS")
print("=" * 75)

print(
    "Total YOLO failures:",
    len(df)
)

print(
    "Pose fallback successful:",
    (
        result["fallback_status"]
        == "pose_box_success"
    ).sum()
)

print(
    "Insufficient pose:",
    (
        result["fallback_status"]
        == "insufficient_pose"
    ).sum()
)


# ============================================================
# METRICS
# ============================================================

successful = result[
    result["fallback_status"]
    == "pose_box_success"
]

print("\n" + "=" * 75)
print("FALLBACK METRICS")
print("=" * 75)

if len(successful) > 0:

    print(
        "Mean Grad-CAM inside pose box:",
        successful[
            "gradcam_inside_pose_box"
        ].mean()
    )

    print(
        "Mean pose inside pose box:",
        successful[
            "pose_inside_box_fraction"
        ].mean()
    )

    print(
        "Mean reliable keypoints:",
        successful[
            "reliable_pose_keypoints"
        ].mean()
    )

    print(
        "Mean pose box area fraction:",
        successful[
            "pose_box_area_fraction"
        ].mean()
    )


# ============================================================
# POTENTIAL RECOVERY
# ============================================================

# Conservative criterion:
# - at least 5 reliable keypoints
# - pose box contains all reliable keypoints
# - Grad-CAM inside region >= 0.50

if len(successful) > 0:

    recovery = successful[
        (
            successful[
                "pose_inside_box_fraction"
            ] >= 0.90
        )
        &
        (
            successful[
                "gradcam_inside_pose_box"
            ] >= 0.50
        )
    ]

    print(
        "\nPotentially recoverable frames:",
        len(recovery)
    )

    print(
        "Potential new success rate:",
        f"{len(recovery) / len(df) * 100:.2f}% "
        "of YOLO failures"
    )

    print(
        "\nPotential total three-way frames:",
        1411 + len(recovery),
        "/ 1669"
    )

    print(
        "Potential total coverage:",
        f"{(1411 + len(recovery)) / 1669 * 100:.2f}%"
    )


print("\nOutput saved to:")

print(OUTPUT_CSV)

print("\nDONE")