
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

MAPPING_FILE = (
    ROOT / "dog_pose_outputs" / "analysis"
    / "exact_gradcam_pose_mapping.csv"
)

GRADCAM_DIR = ROOT / "gradcam_outputs"

OUTPUT_FILE = (
    ROOT / "dog_pose_outputs" / "analysis"
    / "gradcam_pose_image_matches.csv"
)

mapping = pd.read_csv(MAPPING_FILE)

# Find all saved Grad-CAM images
cam_files = sorted(
    p for p in GRADCAM_DIR.rglob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

print("Grad-CAM image files:", len(cam_files))

# Match using the original filename stem
# Example:
# 10319080196_89c41839f2_b.jpg
# 0035_10319080196_89c41839f2_b_angry_1.00.jpg

cam_stems = [p.stem for p in cam_files]

matched_paths = []
for original in mapping["original_filename"]:
    stem = Path(original).stem
    matches = [
        str(path) for path in cam_files
        if stem in path.stem
    ]

    if len(matches) == 1:
        matched_paths.append(matches[0])
    else:
        matched_paths.append(None)

mapping["gradcam_path"] = matched_paths

matched = mapping["gradcam_path"].notna().sum()
unmatched = len(mapping) - matched

print("\nMatched images:", matched)
print("Unmatched images:", unmatched)

if unmatched:
    print("\nUnmatched original filenames:")
    print(
        mapping.loc[
            mapping["gradcam_path"].isna(),
            "original_filename"
        ].head(20).to_string(index=False)
    )

mapping.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("\nSample matches:")
print(
    mapping[
        ["pose_image", "original_filename", "gradcam_path"]
    ].head(10).to_string(index=False)
)