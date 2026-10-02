from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

FAILURE_FILE = (
    ROOT
    / "dog_pose_outputs"
    / "analysis"
    / "dog_detection_failures.csv"
)

OUT = (
    ROOT
    / "dog_pose_outputs"
    / "analysis"
)

OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FAILED FRAMES
# ============================================================

failed = pd.read_csv(FAILURE_FILE)

failed = failed[
    failed["failure_reason"].isin(
        ["no_dog_class", "no_boxes"]
    )
].copy()

print("=" * 60)
print("YOLO FAILURE DIAGNOSTIC")
print("=" * 60)

print("Failed frames:", len(failed))


# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO("yolov8n-seg.pt")


# ============================================================
# COCO CLASS NAMES
# ============================================================

names = model.names


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# PROCESS FAILED FRAMES
# ============================================================

for i, row in failed.iterrows():

    filename = str(row["filename"])

    # Search for original image
    candidates = list(
        (ROOT / "Dataset_test_clean").rglob(filename)
    )

    if not candidates:

        results.append({
            "frame": row["frame"],
            "filename": filename,
            "failure_reason": row["failure_reason"],
            "image_found": False,
            "detected_classes": "",
            "detected_confidences": "",
            "max_detection_confidence": 0.0,
            "largest_box_fraction": 0.0
        })

        continue

    image_path = candidates[0]

    image = cv2.imread(str(image_path))

    if image is None:

        results.append({
            "frame": row["frame"],
            "filename": filename,
            "failure_reason": row["failure_reason"],
            "image_found": False,
            "detected_classes": "",
            "detected_confidences": "",
            "max_detection_confidence": 0.0,
            "largest_box_fraction": 0.0
        })

        continue

    h, w = image.shape[:2]

    image_area = h * w


    # --------------------------------------------------------
    # YOLO prediction
    # --------------------------------------------------------

    pred = model.predict(
        image,
        conf=0.25,
        imgsz=640,
        augment=True,
        verbose=False
    )[0]


    # --------------------------------------------------------
    # Extract detections
    # --------------------------------------------------------

    detected_classes = []
    detected_confidences = []

    largest_box_fraction = 0.0

    if pred.boxes is not None and len(pred.boxes) > 0:

        for j in range(len(pred.boxes)):

            cls_id = int(
                pred.boxes.cls[j].item()
            )

            confidence = float(
                pred.boxes.conf[j].item()
            )

            x1, y1, x2, y2 = (
                pred.boxes.xyxy[j].tolist()
            )

            box_area = max(
                0,
                x2 - x1
            ) * max(
                0,
                y2 - y1
            )

            box_fraction = (
                box_area / image_area
                if image_area > 0
                else 0
            )

            largest_box_fraction = max(
                largest_box_fraction,
                box_fraction
            )

            class_name = names.get(
                cls_id,
                str(cls_id)
            )

            detected_classes.append(
                class_name
            )

            detected_confidences.append(
                round(confidence, 4)
            )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append({

        "frame":
            row["frame"],

        "filename":
            filename,

        "failure_reason":
            row["failure_reason"],

        "image_found":
            True,

        "detected_classes":
            ", ".join(detected_classes),

        "detected_confidences":
            ", ".join(
                map(
                    str,
                    detected_confidences
                )
            ),

        "max_detection_confidence":
            max(
                detected_confidences,
                default=0.0
            ),

        "largest_box_fraction":
            round(
                largest_box_fraction,
                4
            )

    })


    if (len(results)) % 50 == 0:

        print(
            f"Processed {len(results)}/{len(failed)}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

diagnostic = pd.DataFrame(results)

output_file = (
    OUT / "yolo_failure_diagnostic.csv"
)

diagnostic.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("DETECTED CLASS SUMMARY")
print("=" * 60)

class_counter = {}

for classes in diagnostic[
    "detected_classes"
].fillna(""):

    for cls in classes.split(","):

        cls = cls.strip()

        if not cls:
            continue

        class_counter[cls] = (
            class_counter.get(cls, 0) + 1
        )


if class_counter:

    class_summary = pd.Series(
        class_counter
    ).sort_values(
        ascending=False
    )

    print(
        class_summary.to_string()
    )

else:

    print("No objects detected.")


print("\nSaved:")
print(output_file)