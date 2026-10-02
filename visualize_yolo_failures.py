from pathlib import Path
import pandas as pd
import cv2
import numpy as np
import math

ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

FAILURE_CSV = ROOT / "dog_pose_outputs" / "analysis" / "dog_detection_failures.csv"
DIAG_CSV = ROOT / "dog_pose_outputs" / "analysis" / "yolo_failure_diagnostic.csv"
DATASET = ROOT / "Dataset_test_clean"

OUT_DIR = ROOT / "dog_pose_outputs" / "analysis" / "yolo_failure_visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD CSV FILES
# ============================================================

failures = pd.read_csv(FAILURE_CSV)
diag = pd.read_csv(DIAG_CSV)

print("=" * 70)
print("LOADED DATA")
print("=" * 70)

print("Failure rows:", len(failures))
print("Diagnostic rows:", len(diag))


# ============================================================
# MERGE
# ============================================================

diag_cols = [
    "frame",
    "detected_classes",
    "detected_confidences",
    "max_detection_confidence",
    "largest_box_fraction"
]

diag_cols = [
    c for c in diag_cols
    if c in diag.columns
]

df = failures.merge(
    diag[diag_cols],
    on="frame",
    how="left"
)

print("\nMerged columns:")
print(list(df.columns))


# ============================================================
# INDEX ALL DATASET IMAGES
# ============================================================

print("\nIndexing Dataset_test_clean...")

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}

image_files = sorted([
    p for p in DATASET.rglob("*")
    if p.is_file() and p.suffix in extensions
])

print("Total dataset images:", len(image_files))


# ============================================================
# IMPORTANT:
# FRAME NUMBER -> IMAGE INDEX
# ============================================================

def get_image_from_frame(frame):

    try:
        index = int(frame)
    except:
        return None

    if 0 <= index < len(image_files):
        return image_files[index]

    return None


df["image_path"] = df["frame"].apply(
    get_image_from_frame
)


# ============================================================
# CHECK MAPPING
# ============================================================

print("\n" + "=" * 70)
print("FRAME -> IMAGE MAPPING")
print("=" * 70)

found = df["image_path"].notna().sum()
missing = df["image_path"].isna().sum()

print("Images found:", found)
print("Images missing:", missing)

print("\nFirst 10 mappings:")

for _, row in df.head(10).iterrows():

    print(
        "frame =",
        row["frame"],
        "->",
        row["image_path"]
    )


# ============================================================
# MAIN DETECTED CLASS
# ============================================================

def get_main_class(value):

    if pd.isna(value):
        return "unknown"

    text = str(value).strip()

    if not text:
        return "unknown"

    return text.split(",")[0].strip()


df["main_class"] = df[
    "detected_classes"
].apply(get_main_class)


# ============================================================
# CONTACT SHEET FUNCTION
# ============================================================

def create_contact_sheet(
    subset,
    category,
    max_images=10
):

    subset = subset.head(max_images)

    if len(subset) == 0:

        print(
            f"\nNo images for {category}"
        )

        return

    print(
        f"\nCreating {category}: "
        f"{len(subset)} images"
    )

    cells = []

    for _, row in subset.iterrows():

        path = row["image_path"]

        if pd.isna(path):
            continue

        path = Path(path)

        img = cv2.imread(str(path))

        if img is None:
            print(
                "Could not read:",
                path
            )
            continue

        # ----------------------------------------
        # Resize
        # ----------------------------------------

        target_width = 400

        h, w = img.shape[:2]

        scale = target_width / w

        target_height = int(h * scale)

        img = cv2.resize(
            img,
            (target_width, target_height)
        )

        # ----------------------------------------
        # Header
        # ----------------------------------------

        header_height = 70

        canvas = np.zeros(
            (
                target_height + header_height,
                target_width,
                3
            ),
            dtype=np.uint8
        )

        canvas[
            header_height:
        ] = img

        # ----------------------------------------
        # Text
        # ----------------------------------------

        cv2.putText(
            canvas,
            f"YOLO: {category}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            canvas,
            f"frame={row['frame']}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cells.append(canvas)


    if not cells:
        return


    # ========================================================
    # SAME CELL SIZE
    # ========================================================

    cell_width = max(
        x.shape[1]
        for x in cells
    )

    cell_height = max(
        x.shape[0]
        for x in cells
    )

    padded = []

    for img in cells:

        canvas = np.ones(
            (
                cell_height,
                cell_width,
                3
            ),
            dtype=np.uint8
        ) * 255

        canvas[
            :img.shape[0],
            :img.shape[1]
        ] = img

        padded.append(canvas)


    # ========================================================
    # 2-COLUMN GRID
    # ========================================================

    cols = 2

    rows = math.ceil(
        len(padded) / cols
    )

    sheet = np.ones(
        (
            rows * cell_height,
            cols * cell_width,
            3
        ),
        dtype=np.uint8
    ) * 255


    for i, img in enumerate(padded):

        r = i // cols
        c = i % cols

        sheet[
            r * cell_height:(r + 1) * cell_height,
            c * cell_width:(c + 1) * cell_width
        ] = img


    output = OUT_DIR / (
        f"{category}_failures.jpg"
    )

    cv2.imwrite(
        str(output),
        sheet
    )

    print("Saved:", output)


# ============================================================
# CREATE VISUALS
# ============================================================

categories = [
    "cat",
    "bear",
    "person",
    "sheep"
]

for category in categories:

    subset = df[
        df["main_class"]
        .str.lower()
        == category.lower()
    ]

    create_contact_sheet(
        subset,
        category
    )


# ============================================================
# NO-BOXES
# ============================================================

no_boxes = df[
    df["failure_reason"]
    .astype(str)
    .str.lower()
    == "no_boxes"
]

create_contact_sheet(
    no_boxes,
    "no_boxes"
)


# ============================================================
# SAVE MAPPING CSV
# ============================================================

mapping_csv = (
    OUT_DIR /
    "failure_frame_image_mapping.csv"
)

df.to_csv(
    mapping_csv,
    index=False
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print("\nVisual outputs:")
print(OUT_DIR)

print("\nMapping CSV:")
print(mapping_csv)