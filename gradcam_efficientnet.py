import os
import csv
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(
    BASE_DIR,
    "Dataset_test_clean"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "efficientnet_b0_clean_augmented.pth"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "gradcam_outputs"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "gradcam_results.csv"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "angry",
    "curious",
    "happy",
    "sad",
    "sleepy"
]

IMAGE_SIZE = 224


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_name in CLASS_NAMES:

    os.makedirs(
        os.path.join(
            OUTPUT_DIR,
            class_name
        ),
        exist_ok=True
    )


# ============================================================
# TRANSFORM
# SAME AS YOUR EVALUATION SCRIPT
# ============================================================

transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.CenterCrop(224),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("EFFICIENTNET-B0 GRAD-CAM")
print("=" * 70)

print("Device:", DEVICE)
print("Model:", MODEL_PATH)
print("Test directory:", TEST_DIR)

print("\nLoading EfficientNet-B0...")

model = models.efficientnet_b0(
    weights=None
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)
model.eval()


print("Model loaded successfully.")


# ============================================================
# GRAD-CAM TARGET LAYER
# ============================================================

target_layers = [
    model.features[-1]
]

print(
    "Grad-CAM target layer:",
    model.features[-1]
)


# ============================================================
# IMAGE FILE COLLECTION
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


image_files = []

for root, dirs, files in os.walk(TEST_DIR):

    for filename in files:

        if filename.lower().endswith(
            valid_extensions
        ):

            image_files.append(
                os.path.join(
                    root,
                    filename
                )
            )


image_files.sort()


print(
    "\nTest images found:",
    len(image_files)
)


# ============================================================
# CHECK DATASET
# ============================================================

if len(image_files) == 0:

    raise RuntimeError(
        "No images found in Dataset_test_clean."
    )


# ============================================================
# CREATE GRAD-CAM
# ============================================================

cam = GradCAM(
    model=model,
    target_layers=target_layers
)


# ============================================================
# CSV FILE
# ============================================================

csv_file = open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow([
    "image",
    "true_label",
    "predicted_label",
    "confidence",
    "predicted_class_index"
])


# ============================================================
# PROCESS IMAGES
# ============================================================

processed = 0
failed = 0


print("\nStarting Grad-CAM generation...")
print("=" * 70)


for image_index, image_path in enumerate(
    image_files,
    start=1
):

    try:

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        original_image = Image.open(
            image_path
        ).convert("RGB")


        # ----------------------------------------------------
        # ORIGINAL IMAGE FOR VISUALIZATION
        # ----------------------------------------------------

        visualization_image = original_image.resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )

        rgb_image = np.array(
            visualization_image
        ).astype(
            np.float32
        ) / 255.0


        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        input_tensor = transform(
            original_image
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            output = model(
                input_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            predicted_class = output.argmax(
                dim=1
            ).item()

            confidence = probabilities[
                0,
                predicted_class
            ].item()


        predicted_label = CLASS_NAMES[
            predicted_class
        ]


        # ----------------------------------------------------
        # TRUE LABEL
        #
        # Dataset structure expected:
        #
        # Dataset_test_clean/
        #     angry/
        #     curious/
        #     happy/
        #     sad/
        #     sleepy/
        # ----------------------------------------------------

        relative_path = os.path.relpath(
            image_path,
            TEST_DIR
        )

        path_parts = relative_path.split(
            os.sep
        )

        if len(path_parts) >= 2:

            true_label = path_parts[0]

        else:

            true_label = "unknown"


        # ----------------------------------------------------
        # GRAD-CAM TARGET
        # ----------------------------------------------------

        targets = [
            ClassifierOutputTarget(
                predicted_class
            )
        ]


        # ----------------------------------------------------
        # GENERATE CAM
        # ----------------------------------------------------

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )


        grayscale_cam = grayscale_cam[
            0
        ]


        # ----------------------------------------------------
        # CREATE HEATMAP OVERLAY
        # ----------------------------------------------------

        cam_image = show_cam_on_image(
            rgb_image,
            grayscale_cam,
            use_rgb=True
        )


        # ----------------------------------------------------
        # OUTPUT FILE NAME
        # ----------------------------------------------------

        original_filename = os.path.basename(
            image_path
        )

        filename_without_extension = os.path.splitext(
            original_filename
        )[0]


        output_filename = (
            f"{image_index:04d}_"
            f"{filename_without_extension}_"
            f"{predicted_label}_"
            f"{confidence:.2f}.jpg"
        )


        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        output_path = os.path.join(
            OUTPUT_DIR,
            predicted_label,
            output_filename
        )


        # Convert RGB -> BGR for OpenCV

        cam_bgr = cv2.cvtColor(
            cam_image,
            cv2.COLOR_RGB2BGR
        )


        cv2.imwrite(
            output_path,
            cam_bgr
        )


        # ----------------------------------------------------
        # CSV RESULT
        # ----------------------------------------------------

        csv_writer.writerow([
            image_path,
            true_label,
            predicted_label,
            f"{confidence:.6f}",
            predicted_class
        ])


        processed += 1


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (
            image_index == 1
            or image_index % 25 == 0
            or image_index == len(image_files)
        ):

            print(
                f"[{image_index:4d}/{len(image_files)}] "
                f"{predicted_label:<8} "
                f"{confidence * 100:6.2f}%"
            )


    except Exception as e:

        failed += 1

        print(
            f"\nERROR processing:"
            f"\n{image_path}"
            f"\n{e}\n"
        )


# ============================================================
# CLOSE CSV
# ============================================================

csv_file.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("GRAD-CAM COMPLETED")
print("=" * 70)

print(
    "Total images:",
    len(image_files)
)

print(
    "Successfully processed:",
    processed
)

print(
    "Failed:",
    failed
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nCSV file:"
)

print(
    CSV_PATH
)

print("\nSaved folders:")

for class_name in CLASS_NAMES:

    class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    count = len([
        f for f in os.listdir(class_dir)
        if f.lower().endswith(".jpg")
    ])

    print(
        f"  {class_name:<10}: {count} images"
    )

print("\nDone.")