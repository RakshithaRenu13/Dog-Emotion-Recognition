
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm

# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

MODEL_PATH = ROOT / "models" / "efficientnet_b0_clean_augmented.pth"

MAPPING_FILE = (
    ROOT / "dog_pose_outputs" / "analysis"
    / "exact_gradcam_pose_mapping.csv"
)

OUTPUT_DIR = ROOT / "dog_pose_outputs" / "raw_gradcam"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_DIR = OUTPUT_DIR / "maps"
MAP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["angry", "curious", "happy", "sad", "sleepy"]

print("Device:", DEVICE)

# --------------------------------------------------
# 2. LOAD IMAGE MAPPING
# --------------------------------------------------

df = pd.read_csv(MAPPING_FILE)

required_columns = [
    "frame",
    "original_path",
    "original_filename",
    "true_label"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing mapping column: {col}")

if len(df) != 1669:
    print("Warning: Expected 1669 images, found", len(df))

# --------------------------------------------------
# 3. IMAGE PREPROCESSING
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# 4. LOAD TRAINED EFFICIENTNET-B0
# --------------------------------------------------

model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 5
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

print("EfficientNet-B0 loaded successfully.")

# --------------------------------------------------
# 5. GENERATE RAW GRAD-CAM MAPS
# --------------------------------------------------

target_layers = [model.features[-1]]

records = []
errors = []

with GradCAM(
    model=model,
    target_layers=target_layers
) as cam:

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Generating raw Grad-CAM"
    ):
        frame = int(row["frame"])
        image_path = Path(row["original_path"])

        try:
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            image = Image.open(image_path).convert("RGB")
            input_tensor = transform(image).unsqueeze(0).to(DEVICE)

            # Predict the class and generate its Grad-CAM
            with torch.no_grad():
                logits = model(input_tensor)
                predicted_idx = int(
                    torch.argmax(logits, dim=1).item()
                )
                confidence = float(
                    torch.softmax(logits, dim=1)[0, predicted_idx].item()
                )

            targets = [ClassifierOutputTarget(predicted_idx)]

            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]

            # Store the numerical heatmap, not the overlay.
            raw_map = np.asarray(
                grayscale_cam, dtype=np.float32
            )
            raw_map = np.nan_to_num(
                raw_map, nan=0.0, posinf=0.0, neginf=0.0
            )
            raw_map = np.clip(raw_map, 0.0, 1.0)

            map_path = MAP_DIR / f"frame_{frame:05d}.npy"
            np.save(map_path, raw_map)

            predicted_label = CLASS_NAMES[predicted_idx]

            records.append({
                "frame": frame,
                "original_filename": row["original_filename"],
                "true_label": row["true_label"],
                "predicted_label": predicted_label,
                "predicted_class_index": predicted_idx,
                "confidence": confidence,
                "raw_map_path": str(map_path),
                "map_height": raw_map.shape[0],
                "map_width": raw_map.shape[1],
                "map_max": float(raw_map.max()),
                "map_mean": float(raw_map.mean())
            })

        except Exception as e:
            errors.append({
                "frame": frame,
                "filename": row["original_filename"],
                "error": str(e)
            })

# --------------------------------------------------
# 6. SAVE RESULTS
# --------------------------------------------------

results = pd.DataFrame(records)

results_file = OUTPUT_DIR / "raw_gradcam_results.csv"
results.to_csv(results_file, index=False)

errors_file = OUTPUT_DIR / "raw_gradcam_errors.csv"
pd.DataFrame(
    errors, columns=["frame", "filename", "error"]
).to_csv(errors_file, index=False)

print("\n" + "=" * 55)
print("RAW GRAD-CAM GENERATION COMPLETED")
print("=" * 55)
print("Total input images:", len(df))
print("Successfully processed:", len(records))
print("Errors:", len(errors))
print("Output directory:", OUTPUT_DIR)
print("Results CSV:", results_file)

if records:
    print("\nSample results:")
    print(results.head().to_string(index=False))