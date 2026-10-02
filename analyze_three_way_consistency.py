from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

RAW_FILE = ROOT / "dog_pose_outputs" / "raw_gradcam" / "raw_gradcam_results.csv"
POSE_FILE = ROOT / "dog_pose_outputs" / "analysis" / "pose_keypoints.csv"

OUT = ROOT / "dog_pose_outputs" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("Loading input files...")
print("=" * 60)

raw = pd.read_csv(RAW_FILE)
pose = pd.read_csv(POSE_FILE)

print("Raw Grad-CAM records :", len(raw))
print("Pose records         :", len(pose))


# Match Grad-CAM and pose records by frame
data = raw.merge(
    pose,
    on="frame",
    how="inner",
    suffixes=("", "_pose")
)

print("Merged records       :", len(data))


# ============================================================
# LOAD YOLO SEGMENTATION MODEL
# ============================================================

print("\nLoading YOLOv8 segmentation model...")

model = YOLO("yolov8n-seg.pt")

results = []


# ============================================================
# PROCESS EACH FRAME
# ============================================================

for i, row in data.iterrows():

    # --------------------------------------------------------
    # 1. FIND ORIGINAL IMAGE
    # --------------------------------------------------------

    if "original_path" in row and pd.notna(row["original_path"]):

        image_path = Path(str(row["original_path"]))

    else:

        image_path = (
            ROOT
            / "Dataset_test_clean"
            / str(row["true_label"])
            / str(row["original_filename"])
        )

    # If the stored path does not exist, search dataset
    if not image_path.exists():

        candidates = list(
            (ROOT / "Dataset_test_clean").rglob(
                str(row["original_filename"])
            )
        )

        if not candidates:

            results.append({
                "frame": row["frame"],
                "filename": row.get("original_filename", ""),
                "true_label": row.get("true_label", ""),
                "predicted_label": row.get("predicted_label", ""),
                "confidence": row.get("confidence", np.nan),
                "status": "original_missing",
                "failure_reason": "original_missing"
            })

            continue

        image_path = candidates[0]

    # --------------------------------------------------------
    # 2. READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:

        results.append({
            "frame": row["frame"],
            "filename": row.get("original_filename", ""),
            "true_label": row.get("true_label", ""),
            "predicted_label": row.get("predicted_label", ""),
            "confidence": row.get("confidence", np.nan),
            "status": "image_unreadable",
            "failure_reason": "image_unreadable"
        })

        continue

    h, w = image.shape[:2]


    # --------------------------------------------------------
    # 3. FIRST YOLO PREDICTION
    # --------------------------------------------------------

    pred = model.predict(
        image,
        conf=0.25,
        imgsz=640,
        verbose=False
    )[0]


    # --------------------------------------------------------
    # 4. FIND DOGS
    # COCO class 16 = dog
    # --------------------------------------------------------

    dog_ids = []

    has_boxes = (
        pred.boxes is not None
        and len(pred.boxes) > 0
    )

    has_masks = (
        pred.masks is not None
    )

    if has_boxes and has_masks:

        dog_ids = [
            j
            for j, cls in enumerate(pred.boxes.cls.tolist())
            if int(cls) == 16
        ]


    # --------------------------------------------------------
    # 5. RETRY FAILED DETECTIONS
    # SAME CONFIDENCE AS BASELINE
    # --------------------------------------------------------

    if not dog_ids:

        retry = model.predict(
            image,
            conf=0.25,
            imgsz=640,
            augment=True,
            verbose=False
        )[0]

        retry_has_boxes = (
            retry.boxes is not None
            and len(retry.boxes) > 0
        )

        retry_has_masks = (
            retry.masks is not None
        )

        retry_dog_ids = []

        if retry_has_boxes:

            retry_dog_ids = [
                j
                for j, cls in enumerate(
                    retry.boxes.cls.tolist()
                )
                if int(cls) == 16
            ]

        # ----------------------------------------------------
        # Determine exact failure reason
        # ----------------------------------------------------

        if retry_dog_ids and retry_has_masks:

            pred = retry
            dog_ids = retry_dog_ids

        else:

            if not retry_has_boxes:

                failure_reason = "no_boxes"

            elif not retry_dog_ids:

                failure_reason = "no_dog_class"

            elif not retry_has_masks:

                failure_reason = "dog_box_but_no_mask"

            else:

                failure_reason = "unknown"

            results.append({
                "frame": row["frame"],
                "filename": row.get("original_filename", ""),
                "true_label": row.get("true_label", ""),
                "predicted_label": row.get("predicted_label", ""),
                "confidence": row.get("confidence", np.nan),
                "status": "dog_not_detected",
                "failure_reason": failure_reason
            })

            continue


    # --------------------------------------------------------
    # 6. SAFETY CHECK
    # --------------------------------------------------------

    if not dog_ids:

        results.append({
            "frame": row["frame"],
            "filename": row.get("original_filename", ""),
            "true_label": row.get("true_label", ""),
            "predicted_label": row.get("predicted_label", ""),
            "confidence": row.get("confidence", np.nan),
            "status": "dog_not_detected",
            "failure_reason": "unknown"
        })

        continue


    # --------------------------------------------------------
    # 7. SELECT LARGEST DETECTED DOG
    # --------------------------------------------------------

    best = max(
        dog_ids,
        key=lambda j: float(
            (
                pred.boxes.xyxy[j, 2]
                - pred.boxes.xyxy[j, 0]
            )
            *
            (
                pred.boxes.xyxy[j, 3]
                - pred.boxes.xyxy[j, 1]
            )
        )
    )


    # --------------------------------------------------------
    # 8. EXTRACT DOG SEGMENTATION MASK
    # --------------------------------------------------------

    mask = pred.masks.data[best].cpu().numpy()

    mask = cv2.resize(
        mask.astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    mask = mask >= 0.5


    # --------------------------------------------------------
    # 9. LOAD RAW GRAD-CAM MAP
    # --------------------------------------------------------

    if "raw_map_path" not in row or pd.isna(row["raw_map_path"]):

        results.append({
            "frame": row["frame"],
            "filename": row.get("original_filename", ""),
            "true_label": row.get("true_label", ""),
            "predicted_label": row.get("predicted_label", ""),
            "confidence": row.get("confidence", np.nan),
            "status": "gradcam_missing",
            "failure_reason": "gradcam_path_missing"
        })

        continue

    map_path = Path(str(row["raw_map_path"]))

    if not map_path.exists():

        results.append({
            "frame": row["frame"],
            "filename": row.get("original_filename", ""),
            "true_label": row.get("true_label", ""),
            "predicted_label": row.get("predicted_label", ""),
            "confidence": row.get("confidence", np.nan),
            "status": "gradcam_missing",
            "failure_reason": "gradcam_file_missing"
        })

        continue


    # --------------------------------------------------------
    # 10. PROCESS GRAD-CAM
    # --------------------------------------------------------

    heat = np.load(map_path).astype(np.float32)

    heat = np.nan_to_num(
        heat,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    heat = np.maximum(heat, 0)

    heat = cv2.resize(
        heat,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    total_activation = float(heat.sum())

    if total_activation > 0:

        gradcam_inside = float(
            heat[mask].sum() / total_activation
        )

    else:

        gradcam_inside = np.nan


    # --------------------------------------------------------
    # 11. CHECK POSE KEYPOINTS INSIDE DOG MASK
    # --------------------------------------------------------

    keypoint_hits = 0
    keypoint_total = 0

    for col in pose.columns:

        if not col.endswith("_likelihood"):
            continue

        bp = col[:-len("_likelihood")]

        x_col = bp + "_x"
        y_col = bp + "_y"

        if x_col not in pose.columns:
            continue

        if y_col not in pose.columns:
            continue

        # Values from the merged row
        x = row.get(x_col)
        y = row.get(y_col)
        confidence = row.get(col)

        if (
            pd.isna(x)
            or pd.isna(y)
            or pd.isna(confidence)
            or confidence < 0.5
        ):
            continue

        x = int(round(x))
        y = int(round(y))

        if not (0 <= x < w and 0 <= y < h):
            continue

        keypoint_total += 1

        if mask[y, x]:
            keypoint_hits += 1


    # --------------------------------------------------------
    # 12. CALCULATE POSE INSIDE-DOG FRACTION
    # --------------------------------------------------------

    if keypoint_total > 0:

        keypoint_fraction = (
            keypoint_hits / keypoint_total
        )

    else:

        keypoint_fraction = np.nan


    # --------------------------------------------------------
    # 13. SAVE SUCCESSFUL RESULT
    # --------------------------------------------------------

    results.append({

        "frame": row["frame"],

        "filename": row.get(
            "original_filename",
            ""
        ),

        "true_label": row.get(
            "true_label",
            ""
        ),

        "predicted_label": row.get(
            "predicted_label",
            ""
        ),

        "confidence": row.get(
            "confidence",
            np.nan
        ),

        "gradcam_inside_dog_fraction":
            gradcam_inside,

        "reliable_pose_keypoints":
            keypoint_total,

        "pose_keypoints_inside_dog":
            keypoint_hits,

        "pose_inside_dog_fraction":
            keypoint_fraction,

        "status":
            "ok",

        "failure_reason":
            ""

    })


    # --------------------------------------------------------
    # 14. PROGRESS
    # --------------------------------------------------------

    if (i + 1) % 100 == 0:

        print(
            f"Processed {i + 1}/{len(data)}"
        )


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

result_df = pd.DataFrame(results)


# ============================================================
# BASIC FAILURE BREAKDOWN
# ============================================================

print("\n")
print("=" * 60)
print("FAILURE BREAKDOWN")
print("=" * 60)

print(
    result_df["status"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# DETAILED YOLO FAILURE BREAKDOWN
# ============================================================

print("\n")
print("=" * 60)
print("DETAILED YOLO FAILURE BREAKDOWN")
print("=" * 60)

failed = result_df[
    result_df["status"] == "dog_not_detected"
].copy()

if len(failed) > 0 and "failure_reason" in failed.columns:

    print(
        failed["failure_reason"]
        .value_counts(dropna=False)
        .to_string()
    )

else:

    print("No YOLO failures found.")


# ============================================================
# SAVE FRAME-LEVEL RESULTS
# ============================================================

result_file = OUT / "three_way_consistency.csv"

result_df.to_csv(
    result_file,
    index=False
)

print("\nSaved:")
print(result_file)


# ============================================================
# SAVE FAILURE DETAILS SEPARATELY
# ============================================================

failure_file = OUT / "dog_detection_failures.csv"

failed.to_csv(
    failure_file,
    index=False
)

print("Saved:")
print(failure_file)


# ============================================================
# VALID RECORDS
# ============================================================

valid = result_df[
    result_df["status"] == "ok"
].copy()


# ============================================================
# THREE-WAY CONSISTENCY SUMMARY
# ============================================================

summary = pd.DataFrame({

    "metric": [

        "Total input frames",

        "Frames with successful three-way processing",

        "Frames with failed processing",

        "Mean Grad-CAM activation inside dog",

        "Mean reliable pose keypoint fraction inside dog",

        "Frames with at least one reliable pose keypoint"

    ],

    "value": [

        len(data),

        len(valid),

        len(result_df) - len(valid),

        valid[
            "gradcam_inside_dog_fraction"
        ].mean(),

        valid[
            "pose_inside_dog_fraction"
        ].mean(),

        int(
            (
                valid[
                    "reliable_pose_keypoints"
                ] > 0
            ).sum()
        )

    ]

})


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    OUT / "three_way_consistency_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("THREE-WAY CONSISTENCY SUMMARY")
print("=" * 60)

print(
    summary.to_string(index=False)
)

print("\nSaved:")
print(summary_file)

print("\n")
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)