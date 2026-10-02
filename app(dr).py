import streamlit as st
from pathlib import Path

from PIL import Image
import numpy as np
import pandas as pd
import torch
import cv2
import torch.nn as nn
from torchvision import models, transforms

from ultralytics import YOLO
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "Dataset_test_clean"

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(r"C:\Users\acer\dogs-emotion-recognition")

DATASET_DIR = PROJECT_DIR / "Dataset_test_clean"

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

XAI_CSV = (
    PROJECT_DIR
    / "dog_pose_outputs"
    / "analysis"
    / "xai_bodypart"
    / "bodypart_xai_results.csv"
)

YOLO_MODEL = PROJECT_DIR / "yolov8n-seg.pt"
EMOTION_MODEL_PATH = PROJECT_DIR / "models" / "efficientnet_b0_emotion_classifier.pth"
IMAGE_SIZE = 224

# Keep None until the actual 35-keypoint pose model is supplied.
POSE_MODEL = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "angry",
    "curious",
    "happy",
    "sad",
    "sleepy",
]


# ============================================================
# 35 POSE KEYPOINT NAMES
# ============================================================

KEYPOINT_NAMES = [
    "back_base",
    "back_end",
    "back_left_knee",
    "back_left_paw",
    "back_left_thai",
    "back_middle",
    "back_right_knee",
    "back_right_paw",
    "back_right_thai",
    "belly_bottom",
    "body_middle_left",
    "body_middle_right",
    "front_left_knee",
    "front_left_paw",
    "front_left_thai",
    "front_right_knee",
    "front_right_paw",
    "front_right_thai",
    "left_earbase",
    "left_earend",
    "left_eye",
    "lower_jaw",
    "mouth_end_left",
    "mouth_end_right",
    "neck_base",
    "neck_end",
    "nose",
    "right_earbase",
    "right_earend",
    "right_eye",
    "tail_base",
    "tail_end",
    "throat_base",
    "throat_end",
    "upper_jaw",
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Dog Emotion Recognition",
    page_icon="🐕",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🐕 Dog Emotion Recognition</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'AI-based canine emotion recognition with '
    'Grad-CAM, YOLO segmentation, full-body pose '
    'and body-part XAI'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Analysis Settings")

mode = st.sidebar.radio(
    "Choose Input",
    [
        "Test Dataset",
        "New Image",
    ],
)

st.sidebar.markdown("---")

st.sidebar.write("### System")

st.sidebar.write(
    f"Device: `{DEVICE.upper()}`"
)

st.sidebar.write(
    f"Dataset: `{DATASET_DIR.name}`"
)


# ============================================================
# LOAD YOLO SEGMENTATION MODEL
# ============================================================

@st.cache_resource
def load_segmentation_model():

    if not YOLO_MODEL.exists():
        return None

    try:
        return YOLO(str(YOLO_MODEL))
    except Exception as e:
        st.warning(f"Could not load YOLO model: {e}")
        return None


SEG_MODEL = load_segmentation_model()


# ============================================================
# LOAD POSE MODEL
# ============================================================

@st.cache_resource
def load_pose_model():

    if POSE_MODEL is None:
        return None

    pose_path = Path(POSE_MODEL)

    if not pose_path.exists():
        return None

    try:
        return YOLO(str(pose_path))
    except Exception:
        return None


POSE_MODEL_INSTANCE = load_pose_model()


# ============================================================
# LOAD CSV FILES
# ============================================================

@st.cache_data
def load_gradcam_csv():

    if not GRADCAM_CSV.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(GRADCAM_CSV)
    except Exception as e:
        st.warning(f"Grad-CAM CSV loading failed: {e}")
        return pd.DataFrame()


@st.cache_data
def load_pose_csv():

    if not POSE_CSV.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(POSE_CSV)
    except Exception as e:
        st.warning(f"Pose CSV loading failed: {e}")
        return pd.DataFrame()


@st.cache_data
def load_xai_csv():

    if not XAI_CSV.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(XAI_CSV)
    except Exception:
        return pd.DataFrame()


gradcam_df = load_gradcam_csv()
pose_df = load_pose_csv()
xai_df = load_xai_csv()


# ============================================================
# DATASET IMAGE LOADER
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not DATASET_DIR.exists():
    st.error(f"Dataset folder not found: {DATASET_DIR}")
    st.stop()

image_files = sorted(
    p for p in DATASET_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_files:
    st.warning(f"No images found in: {DATASET_DIR}")
else:
    st.success(f"Found {len(image_files)} images")


# ============================================================
# GET FRAME ID
# ============================================================

def get_frame_id(selected_image, images):

    filename = selected_image.name

    # First use Grad-CAM CSV
    if (
        not gradcam_df.empty
        and "original_filename" in gradcam_df.columns
    ):

        matches = gradcam_df[
            gradcam_df["original_filename"]
            .astype(str)
            == filename
        ]

        if not matches.empty:

            try:
                return int(matches.iloc[0]["frame"])
            except Exception:
                pass

    # Fallback
    try:
        return images.index(selected_image)
    except ValueError:
        return None


# ============================================================
# GET GRAD-CAM ROW
# ============================================================

def get_gradcam_row(frame_id):

    if frame_id is None:
        return None

    if gradcam_df.empty:
        return None

    if "frame" not in gradcam_df.columns:
        return None

    rows = gradcam_df[
        gradcam_df["frame"] == frame_id
    ]

    if rows.empty:
        return None

    return rows.iloc[0]


# ============================================================
# STORED PREDICTION
# ============================================================

def get_stored_prediction(frame_id):

    row = get_gradcam_row(frame_id)

    if row is None:
        return None, None

    prediction = row.get(
        "predicted_label",
        None
    )

    confidence = row.get(
        "confidence",
        None
    )

    if pd.isna(prediction):
        prediction = None

    if pd.isna(confidence):
        confidence = None

    if confidence is not None:

        try:
            confidence = float(confidence)

            if confidence > 1:
                confidence /= 100.0

        except Exception:
            confidence = None

    return prediction, confidence


# ============================================================
# RESOLVE GRAD-CAM FILE PATH
# ============================================================

def resolve_gradcam_path(raw_path):

    if raw_path is None:
        return None

    try:
        if pd.isna(raw_path):
            return None
    except Exception:
        pass

    raw_path = Path(str(raw_path))

    # Direct path
    if raw_path.exists():
        return raw_path

    # Project-relative path
    candidate = PROJECT_DIR / raw_path

    if candidate.exists():
        return candidate

    # CSV folder + filename
    candidate = GRADCAM_CSV.parent / raw_path

    if candidate.exists():
        return candidate

    # Same filename inside raw_gradcam
    candidate = (
        PROJECT_DIR
        / "dog_pose_outputs"
        / "raw_gradcam"
        / raw_path.name
    )

    if candidate.exists():
        return candidate

    return None


# ============================================================
# LOAD PRECOMPUTED GRAD-CAM
# ============================================================

def load_precomputed_gradcam(frame_id, image):

    row = get_gradcam_row(frame_id)

    if row is None:
        return None, None, "No Grad-CAM CSV row found."

    raw_path = row.get(
        "raw_map_path",
        None
    )

    raw_path = resolve_gradcam_path(raw_path)

    if raw_path is None:
        return (
            None,
            None,
            "Grad-CAM .npy file could not be located."
        )

    try:

        cam = np.load(str(raw_path))

        cam = np.asarray(
            cam,
            dtype=np.float32
        )

        cam = np.squeeze(cam)

        if cam.ndim != 2:

            return (
                None,
                None,
                f"Invalid Grad-CAM shape: {cam.shape}"
            )

        cam_min = float(cam.min())
        cam_max = float(cam.max())

        if cam_max > cam_min:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(cam)

        img = np.array(
            image.convert("RGB")
        )

        h, w = img.shape[:2]

        cam_resized = cv2.resize(
            cam,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

        heatmap = cv2.applyColorMap(
            np.uint8(
                np.clip(
                    cam_resized,
                    0,
                    1
                ) * 255
            ),
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        overlay = cv2.addWeighted(
            img,
            0.55,
            heatmap,
            0.45,
            0
        )

        return (
            overlay,
            cam_resized,
            f"Loaded: {raw_path.name}"
        )

    except Exception as e:

        return (
            None,
            None,
            f"Grad-CAM loading error: {e}"
        )


# ============================================================
# TRAINED EMOTION MODEL + GRAD-CAM
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@st.cache_resource
def load_emotion_model():
    """Load the same EfficientNet-B0 architecture used by the project."""
    if not EMOTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Emotion checkpoint not found: {EMOTION_MODEL_PATH}"
        )

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(CLASS_NAMES)
    )

    try:
        checkpoint = torch.load(
            str(EMOTION_MODEL_PATH),
            map_location=DEVICE,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            str(EMOTION_MODEL_PATH), map_location=DEVICE
        )

    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint)
        # Handle checkpoints saved from DataParallel.
        if state and all(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def make_gradcam(model, input_tensor, class_index, original_image):
    """Compute Grad-CAM for the predicted class at EfficientNet's final feature block."""
    activations = []
    gradients = []
    target_layer = model.features[-1]

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        input_tensor = input_tensor.to(DEVICE)
        input_tensor.requires_grad_(True)

        logits = model(input_tensor)
        score = logits[:, class_index].sum()
        score.backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture model tensors.")

        feature = activations[0].detach()
        gradient = gradients[0].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam.cpu().numpy().astype(np.float32)
        cam -= float(cam.min())
        maximum = float(cam.max())
        if maximum > 1e-8:
            cam /= maximum
        else:
            cam = np.zeros_like(cam)

        original = np.asarray(original_image.convert("RGB"))
        height, width = original.shape[:2]
        cam_resized = cv2.resize(
            cam, (width, height), interpolation=cv2.INTER_LINEAR
        )
        heatmap = cv2.applyColorMap(
            np.uint8(np.clip(cam_resized, 0, 1) * 255),
            cv2.COLORMAP_JET,
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.55, heatmap, 0.45, 0)
        return overlay, cam_resized
    finally:
        forward_handle.remove()
        backward_handle.remove()


def predict_uploaded_image(image):
    """Return real model prediction, probabilities and class-specific Grad-CAM."""
    model = load_emotion_model()
    tensor = INFERENCE_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    # Grad-CAM needs gradients, so do not wrap this block in torch.no_grad().
    model.zero_grad(set_to_none=True)
    tensor.requires_grad_(True)
    logits = model(tensor)
    probabilities = torch.softmax(logits.detach(), dim=1)[0].cpu().numpy()
    class_index = int(np.argmax(probabilities))
    prediction = CLASS_NAMES[class_index]
    confidence = float(probabilities[class_index])

    gradcam_image, cam = make_gradcam(
        model, tensor.detach(), class_index, image
    )
    probability_table = pd.DataFrame({
        "Emotion": CLASS_NAMES,
        "Probability (%)": (probabilities * 100).round(2),
    }).sort_values("Probability (%)", ascending=False, ignore_index=True)

    return prediction, confidence, probability_table, gradcam_image, cam


# ============================================================
# YOLO DOG SEGMENTATION
# ============================================================

def get_dog_segmentation(image):

    if SEG_MODEL is None:

        return (
            None,
            None,
            np.array(image.convert("RGB")),
            False,
            "YOLO segmentation model unavailable."
        )

    img = np.array(
        image.convert("RGB")
    )

    h, w = img.shape[:2]

    # Start with the normal threshold.
    # Only lower it if no dog is found.
    confidence_levels = [
        0.25,
        0.10,
        0.05
    ]

    best_result = None
    best_index = None
    best_box = None
    best_area = 0
    best_confidence = 0.0

    for conf_threshold in confidence_levels:

        try:

            results = SEG_MODEL.predict(
                source=img,
                conf=conf_threshold,
                imgsz=640,
                device=DEVICE,
                verbose=False
            )

        except Exception as e:

            continue

        if not results:
            continue

        result = results[0]

        if (
            result.boxes is None
            or len(result.boxes) == 0
        ):
            continue

        boxes = result.boxes

        for i in range(len(boxes)):

            try:

                cls_id = int(
                    boxes.cls[i].item()
                )

                # COCO class 16 = dog
                if cls_id != 16:
                    continue

                confidence = float(
                    boxes.conf[i].item()
                )

                coords = (
                    boxes.xyxy[i]
                    .detach()
                    .cpu()
                    .numpy()
                )

                x1, y1, x2, y2 = coords

                x1 = max(
                    0,
                    min(w - 1, int(x1))
                )

                y1 = max(
                    0,
                    min(h - 1, int(y1))
                )

                x2 = max(
                    0,
                    min(w - 1, int(x2))
                )

                y2 = max(
                    0,
                    min(h - 1, int(y2))
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                area = (
                    (x2 - x1)
                    * (y2 - y1)
                )

                if area > best_area:

                    best_area = area
                    best_result = result
                    best_index = i

                    best_box = (
                        x1,
                        y1,
                        x2,
                        y2
                    )

                    best_confidence = confidence

            except Exception:
                continue

        # Stop as soon as a dog is found.
        if best_box is not None:
            break

    # ========================================================
    # NO DOG
    # ========================================================

    if best_box is None:

        return (
            None,
            None,
            img.copy(),
            False,
            "YOLO did not detect a dog."
        )

    x1, y1, x2, y2 = best_box

    # ========================================================
    # CREATE MASK
    # ========================================================

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    mask_created = False

    try:

        if (
            best_result is not None
            and best_result.masks is not None
        ):

            all_masks = (
                best_result.masks.data
                .detach()
                .cpu()
                .numpy()
            )

            if (
                best_index is not None
                and best_index < len(all_masks)
            ):

                raw_mask = all_masks[
                    best_index
                ]

                raw_mask = cv2.resize(
                    raw_mask,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

                mask = (
                    raw_mask > 0.5
                ).astype(np.uint8)

                if mask.sum() > 0:
                    mask_created = True

    except Exception:
        mask_created = False

    # ========================================================
    # BOUNDING BOX FALLBACK
    # ========================================================

    if not mask_created:

        mask[
            y1:y2 + 1,
            x1:x2 + 1
        ] = 1

    # ========================================================
    # DRAW
    # ========================================================

    output = img.copy()

    overlay = output.copy()

    dog_pixels = mask > 0

    if np.any(dog_pixels):

        overlay[dog_pixels] = (
            0.55 * overlay[dog_pixels]
            + 0.45 * np.array(
                [0, 255, 0],
                dtype=np.float32
            )
        ).astype(np.uint8)

    output = overlay

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2
    )

    label_text = (
        f"DOG {best_confidence * 100:.1f}%"
    )

    cv2.putText(
        output,
        label_text,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return (
        mask,
        best_box,
        output,
        True,
        f"Dog detected with {best_confidence * 100:.1f}% confidence."
    )


# ============================================================
# POSE CSV HELPERS
# ============================================================

def find_pose_column(row, keypoint, suffixes):

    for suffix in suffixes:

        candidates = [
            f"{keypoint}_{suffix}",
            f"{keypoint}.{suffix}",
            f"{keypoint}{suffix}",
        ]

        for column in candidates:

            if column in row.index:
                return row[column]

    return np.nan


def extract_pose_from_csv_row(row):

    points = {}

    for keypoint in KEYPOINT_NAMES:

        x = find_pose_column(
            row,
            keypoint,
            ["x", "X"]
        )

        y = find_pose_column(
            row,
            keypoint,
            ["y", "Y"]
        )

        confidence = find_pose_column(
            row,
            keypoint,
            [
                "likelihood",
                "confidence",
                "conf",
                "score",
                "probability"
            ]
        )

        try:
            x = float(x)
            y = float(y)
        except Exception:
            continue

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 1.0

        if (
            np.isfinite(x)
            and np.isfinite(y)
        ):

            points[keypoint] = {
                "x": x,
                "y": y,
                "confidence": confidence
            }

    return points


# ============================================================
# FILTER POSE USING DOG MASK
# ============================================================

def filter_named_pose_points(
    points,
    dog_mask,
    dog_box,
    confidence_threshold=0.50
):

    if not points:
        return {}

    filtered = {}

    if dog_mask is not None:

        h, w = dog_mask.shape[:2]

    else:

        h, w = None, None

    for name, point in points.items():

        x = float(point["x"])
        y = float(point["y"])

        confidence = float(
            point["confidence"]
        )

        if confidence < confidence_threshold:
            continue

        if not (
            np.isfinite(x)
            and np.isfinite(y)
        ):
            continue

        px = int(round(x))
        py = int(round(y))

        if h is not None:

            if (
                px < 0
                or px >= w
                or py < 0
                or py >= h
            ):
                continue

        inside_dog = False

        # ----------------------------------------------------
        # Actual segmentation mask
        # ----------------------------------------------------

        if dog_mask is not None:

            if (
                0 <= py < dog_mask.shape[0]
                and 0 <= px < dog_mask.shape[1]
            ):

                if dog_mask[py, px] > 0:
                    inside_dog = True

        # ----------------------------------------------------
        # Small neighborhood fallback
        # ----------------------------------------------------

        if not inside_dog and dog_mask is not None:

            y1 = max(0, py - 2)
            y2 = min(
                dog_mask.shape[0],
                py + 3
            )

            x1 = max(0, px - 2)
            x2 = min(
                dog_mask.shape[1],
                px + 3
            )

            if np.any(
                dog_mask[y1:y2, x1:x2] > 0
            ):
                inside_dog = True

        # ----------------------------------------------------
        # Bounding-box fallback
        # ----------------------------------------------------

        if (
            not inside_dog
            and dog_box is not None
        ):

            x1, y1, x2, y2 = dog_box

            if (
                x1 <= px <= x2
                and y1 <= py <= y2
            ):
                inside_dog = True

        if inside_dog:

            filtered[name] = {
                "x": x,
                "y": y,
                "confidence": confidence
            }

    return filtered


# ============================================================
# POSE CONNECTIONS
# ============================================================

POSE_CONNECTIONS = [

    ("nose", "upper_jaw"),
    ("upper_jaw", "lower_jaw"),
    ("lower_jaw", "neck_end"),
    ("neck_end", "neck_base"),

    ("neck_base", "back_base"),
    ("back_base", "back_middle"),
    ("back_middle", "back_end"),

    ("back_end", "tail_base"),
    ("tail_base", "tail_end"),

    ("left_earbase", "left_earend"),
    ("right_earbase", "right_earend"),

    ("left_eye", "nose"),
    ("right_eye", "nose"),

    ("mouth_end_left", "lower_jaw"),
    ("mouth_end_right", "lower_jaw"),

    ("front_left_thai", "front_left_knee"),
    ("front_left_knee", "front_left_paw"),

    ("front_right_thai", "front_right_knee"),
    ("front_right_knee", "front_right_paw"),

    ("back_left_thai", "back_left_knee"),
    ("back_left_knee", "back_left_paw"),

    ("back_right_thai", "back_right_knee"),
    ("back_right_knee", "back_right_paw"),
]


# ============================================================
# DRAW POSE
# ============================================================

def draw_named_pose(image, points):

    output = np.array(
        image.convert("RGB")
    ).copy()

    # Skeleton
    for a, b in POSE_CONNECTIONS:

        if (
            a not in points
            or b not in points
        ):
            continue

        p1 = (
            int(points[a]["x"]),
            int(points[a]["y"])
        )

        p2 = (
            int(points[b]["x"]),
            int(points[b]["y"])
        )

        cv2.line(
            output,
            p1,
            p2,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Keypoints
    for name, point in points.items():

        x = int(point["x"])
        y = int(point["y"])

        cv2.circle(
            output,
            (x, y),
            5,
            (255, 0, 0),
            -1
        )

    return output


# ============================================================
# GET STORED POSE
# ============================================================

def get_stored_pose_for_frame(
    frame_id,
    image,
    dog_mask,
    dog_box
):

    if pose_df.empty:

        return (
            None,
            {},
            "Pose CSV not available."
        )

    if "frame" not in pose_df.columns:

        return (
            None,
            {},
            "Pose CSV has no frame column."
        )

    rows = pose_df[
        pose_df["frame"] == frame_id
    ]

    if rows.empty:

        return (
            None,
            {},
            f"No stored pose for frame {frame_id}."
        )

    row = rows.iloc[0]

    raw_points = extract_pose_from_csv_row(row)

    if not raw_points:

        return (
            None,
            {},
            "Could not read pose keypoints from CSV."
        )

    filtered_points = filter_named_pose_points(
        raw_points,
        dog_mask,
        dog_box,
        confidence_threshold=0.50
    )

    if not filtered_points:

        return (
            None,
            {},
            "No reliable pose keypoints remained inside the dog."
        )

    pose_image = draw_named_pose(
        image,
        filtered_points
    )

    return (
        pose_image,
        filtered_points,
        f"{len(filtered_points)} pose keypoints kept inside dog."
    )


# ============================================================
# LIVE POSE
# ============================================================

def run_live_pose(image, dog_mask):

    if POSE_MODEL_INSTANCE is None:

        return (
            None,
            {},
            False,
            "Live 35-keypoint pose model is not configured."
        )

    if dog_mask is None:

        return (
            None,
            {},
            False,
            "Dog segmentation is unavailable."
        )

    img = np.array(
        image.convert("RGB")
    )

    try:

        results = POSE_MODEL_INSTANCE.predict(
            source=img,
            conf=0.25,
            imgsz=640,
            device=DEVICE,
            verbose=False
        )

    except Exception as e:

        return (
            None,
            {},
            False,
            f"Pose inference error: {e}"
        )

    if not results:

        return (
            None,
            {},
            False,
            "No pose result."
        )

    result = results[0]

    if result.keypoints is None:

        return (
            None,
            {},
            False,
            "No keypoints detected."
        )

    xy = (
        result.keypoints.xy
        .detach()
        .cpu()
        .numpy()
    )

    if result.keypoints.conf is not None:

        conf = (
            result.keypoints.conf
            .detach()
            .cpu()
            .numpy()
        )

    else:

        conf = np.ones(
            xy.shape[:2],
            dtype=float
        )

    best_points = None
    best_count = -1

    for detection_idx in range(len(xy)):

        points = xy[detection_idx]
        confidences = conf[detection_idx]

        filtered = []

        for i, point in enumerate(points):

            x = float(point[0])
            y = float(point[1])

            c = float(
                confidences[i]
            )

            if c < 0.50:
                continue

            px = int(round(x))
            py = int(round(y))

            if (
                px < 0
                or py < 0
                or px >= dog_mask.shape[1]
                or py >= dog_mask.shape[0]
            ):
                continue

            if dog_mask[py, px] == 0:
                continue

            filtered.append(
                {
                    "x": x,
                    "y": y,
                    "confidence": c,
                    "index": i
                }
            )

        if len(filtered) > best_count:

            best_count = len(filtered)
            best_points = filtered

    if not best_points:

        return (
            None,
            {},
            False,
            "No pose keypoints remained inside dog."
        )

    output = np.array(
        image.convert("RGB")
    ).copy()

    for point in best_points:

        x = int(point["x"])
        y = int(point["y"])

        cv2.circle(
            output,
            (x, y),
            5,
            (255, 0, 0),
            -1
        )

    return (
        output,
        {
            str(p["index"]): p
            for p in best_points
        },
        True,
        f"{len(best_points)} keypoints kept inside dog."
    )


# ============================================================
# BODY REGIONS
# ============================================================

BODY_REGIONS = {

    "head": [
        "nose",
        "left_eye",
        "right_eye",
        "neck_base",
        "neck_end",
        "upper_jaw",
        "lower_jaw"
    ],

    "ears": [
        "left_earbase",
        "left_earend",
        "right_earbase",
        "right_earend"
    ],

    "mouth": [
        "upper_jaw",
        "lower_jaw",
        "mouth_end_left",
        "mouth_end_right"
    ],

    "body": [
        "body_middle_left",
        "body_middle_right",
        "back_middle",
        "belly_bottom",
        "throat_base",
        "throat_end"
    ],

    "front_legs": [
        "front_left_thai",
        "front_left_knee",
        "front_left_paw",
        "front_right_thai",
        "front_right_knee",
        "front_right_paw"
    ],

    "hind_legs": [
        "back_left_thai",
        "back_left_knee",
        "back_left_paw",
        "back_right_thai",
        "back_right_knee",
        "back_right_paw"
    ],

    "tail": [
        "tail_base",
        "tail_end",
        "back_end"
    ]
}


# ============================================================
# BODY PART XAI
# ============================================================

def calculate_bodypart_xai(
    cam,
    pose_points,
    original_shape
):

    if cam is None:
        return {}

    if not pose_points:
        return {}

    cam = np.asarray(
        cam,
        dtype=np.float32
    )

    cam_h, cam_w = cam.shape[:2]

    original_h, original_w = original_shape[:2]

    overall_mean = float(
        np.mean(cam)
    )

    if overall_mean <= 0:
        overall_mean = 1e-8

    results = {}

    for region, keypoints in BODY_REGIONS.items():

        values = []

        for name in keypoints:

            if name not in pose_points:
                continue

            original_x = float(
                pose_points[name]["x"]
            )

            original_y = float(
                pose_points[name]["y"]
            )

            # IMPORTANT:
            # Pose coordinates are original-image coordinates.
            # Convert them to CAM coordinates.

            px = int(
                round(
                    original_x
                    * cam_w
                    / original_w
                )
            )

            py = int(
                round(
                    original_y
                    * cam_h
                    / original_h
                )
            )

            if (
                px < 0
                or px >= cam_w
                or py < 0
                or py >= cam_h
            ):
                continue

            radius = 8

            x1 = max(
                0,
                px - radius
            )

            x2 = min(
                cam_w,
                px + radius + 1
            )

            y1 = max(
                0,
                py - radius
            )

            y2 = min(
                cam_h,
                py + radius + 1
            )

            patch = cam[
                y1:y2,
                x1:x2
            ]

            if patch.size > 0:

                values.append(
                    float(
                        np.mean(patch)
                    )
                )

        if values:

            region_mean = float(
                np.mean(values)
            )

            relative_score = (
                region_mean
                / overall_mean
                * 100.0
            )

            results[region] = {
                "mean_activation": region_mean,
                "relative_score": relative_score,
                "keypoints_used": len(values)
            }

    return results


# ============================================================
# DISPLAY BODY-PART XAI
# ============================================================

def show_live_bodypart_xai(
    cam,
    pose_points,
    prediction,
    confidence,
    original_image
):

    if cam is None:

        st.info(
            "Grad-CAM is unavailable."
        )

        return

    if not pose_points:

        st.info(
            "No filtered pose points are available "
            "for body-part XAI."
        )

        return

    results = calculate_bodypart_xai(
        cam,
        pose_points,
        np.array(original_image).shape
    )

    if not results:

        st.warning(
            "No body-part XAI values could be calculated."
        )

        return

    rows = []

    for region, values in results.items():

        rows.append(
            {
                "Body Region": region.title(),
                "Relative Attention Score":
                    values["relative_score"],
                "Keypoints Used":
                    values["keypoints_used"]
            }
        )

    df = pd.DataFrame(rows)

    df = df.sort_values(
        "Relative Attention Score",
        ascending=False
    )

    dominant = df.iloc[0]

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Emotion",
            str(prediction).upper()
        )

    with c2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with c3:

        st.metric(
            "Highest Attention Region",
            dominant["Body Region"]
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        df.set_index(
            "Body Region"
        )["Relative Attention Score"]
    )

    st.caption(
        "These scores represent Grad-CAM activation "
        "around pose-defined body regions. Regions "
        "can overlap, so the values are not percentages "
        "of the final emotion decision and do not need "
        "to sum to 100%."
    )


# ============================================================
# OLD XAI CSV FALLBACK
# ============================================================

def show_xai_explanation(
    frame_id,
    predicted_label,
    confidence
):

    if xai_df.empty:

        st.warning(
            "Body-part XAI CSV was not found."
        )

        return

    if frame_id is None:
        return

    if "frame" not in xai_df.columns:

        st.error(
            "XAI CSV does not contain a frame column."
        )

        return

    rows = xai_df[
        xai_df["frame"] == frame_id
    ]

    if rows.empty:

        st.warning(
            f"No stored XAI result for frame {frame_id}."
        )

        return

    row = rows.iloc[0]

    st.info(
        "The emotion classifier determines the emotion. "
        "Grad-CAM indicates where the classifier focused. "
        "Pose estimation identifies anatomical keypoints. "
        "Body-part XAI measures Grad-CAM activation around "
        "those pose-defined regions."
    )

    dominant = row.get(
        "dominant_body_region",
        "unknown"
    )

    if pd.isna(dominant):
        dominant = "unknown"

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Emotion",
            predicted_label.upper()
        )

    with c2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with c3:

        st.metric(
            "Dominant Region",
            str(dominant)
            .replace("_", " ")
            .title()
        )

    regions = {

        "Head":
            row.get(
                "head_percent",
                np.nan
            ),

        "Ears":
            row.get(
                "ears_percent",
                np.nan
            ),

        "Mouth":
            row.get(
                "mouth_percent",
                np.nan
            ),

        "Body":
            row.get(
                "body_percent",
                np.nan
            ),

        "Front Legs":
            row.get(
                "front_legs_percent",
                np.nan
            ),

        "Hind Legs":
            row.get(
                "hind_legs_percent",
                np.nan
            ),

        "Tail":
            row.get(
                "tail_percent",
                np.nan
            )
    }

    region_df = pd.DataFrame(
        {
            "Body Region":
                list(regions.keys()),

            "Attention Score":
                list(regions.values())
        }
    )

    region_df = region_df.dropna()

    if not region_df.empty:

        region_df = region_df[
            np.isfinite(
                region_df[
                    "Attention Score"
                ]
            )
        ]

    if not region_df.empty:

        region_df = region_df.sort_values(
            "Attention Score",
            ascending=False
        )

        st.dataframe(
            region_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            region_df.set_index(
                "Body Region"
            )["Attention Score"]
        )

    st.caption(
        "These measurements describe where Grad-CAM "
        "activation was concentrated. Body regions may "
        "overlap and therefore should not be interpreted "
        "as percentages that must sum to 100%."
    )


# ============================================================
# TEST DATASET MODE
# ============================================================

if mode == "Test Dataset":

    st.header("🧪 Test Dataset")

    images = get_dataset_images()

    if len(images) == 0:

        st.error(
            f"No images found in:\n{DATASET_DIR}"
        )

        st.stop()

    st.write(
        f"**{len(images)} test images available**"
    )

    available_classes = sorted(
        {
            p.parent.name
            for p in images
        }
    )

    selected_class = st.selectbox(
        "Select Emotion Class",
        available_classes
    )

    class_images = [
        p
        for p in images
        if p.parent.name == selected_class
    ]

    if not class_images:

        st.warning(
            "No images found for this class."
        )

        st.stop()

    selected_image = st.selectbox(
        "Select Test Image",
        class_images,
        format_func=lambda x: x.name
    )

    st.write(
        f"Selected image: `{selected_image.name}`"
    )

    if st.button(
        "🔍 Analyze Test Image",
        type="primary",
        use_container_width=True
    ):

        image = Image.open(
            selected_image
        ).convert("RGB")

        frame_id = get_frame_id(
            selected_image,
            images
        )

        # ----------------------------------------------------
        # STORED PREDICTION
        # ----------------------------------------------------

        prediction, confidence = (
            get_stored_prediction(frame_id)
        )

        # IMPORTANT:
        # Do NOT use fake prediction for test data.

        if (
            prediction is None
            or confidence is None
        ):

            st.error(
                f"No stored prediction found for "
                f"frame {frame_id}. "
                f"Check raw_gradcam_results.csv."
            )

            st.stop()

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        (
            dog_mask,
            dog_box,
            segmentation_image,
            segmentation_success,
            segmentation_message
        ) = get_dog_segmentation(image)

        # ----------------------------------------------------
        # STORED POSE
        # ----------------------------------------------------

        (
            pose_image,
            filtered_points,
            pose_message
        ) = get_stored_pose_for_frame(
            frame_id,
            image,
            dog_mask,
            dog_box
        )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        (
            gradcam_image,
            cam,
            gradcam_message
        ) = load_precomputed_gradcam(
            frame_id,
            image
        )

        # ----------------------------------------------------
        # STORE RESULTS
        # ----------------------------------------------------

        st.session_state["image"] = image

        st.session_state["prediction"] = str(
            prediction
        )
        st.session_state["probability_table"] = None
        st.session_state["analysis_source"] = "test_dataset"

        st.session_state["confidence"] = float(
            confidence
        )

        st.session_state["true_label"] = (
            selected_class
        )

        st.session_state["frame_id"] = frame_id

        st.session_state["filename"] = (
            selected_image.name
        )

        st.session_state[
            "live_segmentation"
        ] = segmentation_image

        st.session_state[
            "live_dog_mask"
        ] = dog_mask

        st.session_state[
            "live_dog_box"
        ] = dog_box

        st.session_state[
            "live_segmentation_success"
        ] = segmentation_success

        st.session_state[
            "live_segmentation_message"
        ] = segmentation_message

        st.session_state[
            "live_pose"
        ] = pose_image

        st.session_state[
            "live_filtered_points"
        ] = filtered_points

        st.session_state[
            "live_pose_success"
        ] = bool(filtered_points)

        st.session_state[
            "live_pose_message"
        ] = pose_message

        st.session_state[
            "live_gradcam"
        ] = gradcam_image

        st.session_state[
            "live_cam"
        ] = cam

        st.session_state[
            "gradcam_message"
        ] = gradcam_message


# ============================================================
# NEW IMAGE MODE
# ============================================================

else:
    st.header("📤 New Image Prediction")
    st.write(
        "Upload a new image. The trained EfficientNet-B0 model will "
        "predict one of the five project emotion classes."
    )

    uploaded_file = st.file_uploader(
        "Upload a dog image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="new_image_uploader",
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=500)

            if st.button(
                "🔍 Analyze New Image",
                type="primary",
                use_container_width=True,
                key="analyze_new_image",
            ):
                with st.spinner("Running emotion prediction, Grad-CAM and dog segmentation..."):
                    prediction, confidence, probability_table, gradcam_image, cam = (
                        predict_uploaded_image(image)
                    )

                    (
                        dog_mask,
                        dog_box,
                        segmentation_image,
                        segmentation_success,
                        segmentation_message,
                    ) = get_dog_segmentation(image)

                    (
                        pose_image,
                        filtered_points,
                        pose_success,
                        pose_message,
                    ) = run_live_pose(image, dog_mask)

                # Clear stale dataset/new-upload results and store this image's results.
                st.session_state["image"] = image
                st.session_state["prediction"] = prediction
                st.session_state["confidence"] = confidence
                st.session_state["probability_table"] = probability_table
                st.session_state["true_label"] = None
                st.session_state["frame_id"] = None
                st.session_state["filename"] = uploaded_file.name
                st.session_state["live_segmentation"] = segmentation_image
                st.session_state["live_dog_mask"] = dog_mask
                st.session_state["live_dog_box"] = dog_box
                st.session_state["live_segmentation_success"] = segmentation_success
                st.session_state["live_segmentation_message"] = segmentation_message
                st.session_state["live_pose"] = pose_image
                st.session_state["live_pose_success"] = pose_success
                st.session_state["live_pose_message"] = pose_message
                st.session_state["live_filtered_points"] = filtered_points
                st.session_state["live_gradcam"] = gradcam_image
                st.session_state["live_cam"] = cam
                st.session_state["gradcam_message"] = (
                    "Grad-CAM generated for the predicted emotion."
                )
                st.session_state["analysis_source"] = "new_image"

        except Exception as e:
            st.error(f"Could not read or analyze the uploaded image: {e}")
            st.exception(e)


# ============================================================
# RESULTS
# ============================================================

if (
    "prediction" in st.session_state
    and st.session_state["prediction"] is not None
):

    image = st.session_state["image"]

    prediction = st.session_state[
        "prediction"
    ]

    confidence = st.session_state[
        "confidence"
    ]

    true_label = st.session_state.get(
        "true_label"
    )

    frame_id = st.session_state.get(
        "frame_id"
    )

    filename = st.session_state.get(
        "filename",
        "Unknown"
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    st.markdown("---")

    st.header("🎯 Emotion Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Predicted Emotion",
            str(prediction).upper()
        )

    with col2:

        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )

    with col3:

        if true_label is not None:

            result = (
                "Correct"
                if str(prediction).lower()
                == str(true_label).lower()
                else "Incorrect"
            )

            st.metric(
                "Prediction Result",
                result
            )

        else:

            st.metric(
                "Input",
                "New Image"
            )

    if true_label is not None:

        st.info(
            f"Ground Truth: **{true_label.upper()}**"
        )

    probability_table = st.session_state.get("probability_table")
    if probability_table is not None and true_label is None:
        st.subheader("Class probabilities")
        st.dataframe(
            probability_table,
            use_container_width=True,
            hide_index=True,
        )
        if confidence < 0.50:
            st.warning(
                "The model has relatively low confidence for this image. "
                "Treat the predicted emotion as uncertain."
            )

    if frame_id is not None:

        st.caption(
            f"Dataset frame: {frame_id} | "
            f"File: {filename}"
        )

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    st.markdown("---")

    st.header("🔎 Explainability Analysis")

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            use_container_width=True
        )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    with col2:

        st.subheader("Grad-CAM")

        gradcam = st.session_state.get(
            "live_gradcam"
        )

        if gradcam is not None:

            st.image(
                gradcam,
                use_container_width=True
            )

            message = st.session_state.get(
                "gradcam_message",
                ""
            )

            if message:
                st.caption(message)

        else:

            st.warning(
                st.session_state.get(
                    "gradcam_message",
                    "Grad-CAM unavailable."
                )
            )

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    with col3:

        st.subheader("Dog Segmentation")

        segmentation = st.session_state.get(
            "live_segmentation"
        )

        segmentation_success = (
            st.session_state.get(
                "live_segmentation_success",
                False
            )
        )

        if segmentation is not None:

            st.image(
                segmentation,
                use_container_width=True
            )

            if segmentation_success:

                st.success(
                    st.session_state.get(
                        "live_segmentation_message",
                        "Dog detected."
                    )
                )

            else:

                st.warning(
                    st.session_state.get(
                        "live_segmentation_message",
                        "Dog was not detected."
                    )
                )

        else:

            st.warning(
                "Segmentation unavailable."
            )

    # ========================================================
    # FULL BODY POSE
    # ========================================================

    st.markdown("---")

    st.header("🦴 Full-Body Pose")

    pose_image = st.session_state.get(
        "live_pose"
    )

    pose_success = st.session_state.get(
        "live_pose_success",
        False
    )

    pose_message = st.session_state.get(
        "live_pose_message",
        ""
    )

    filtered_points = st.session_state.get(
        "live_filtered_points",
        {}
    )

    if pose_image is not None:

        st.image(
            pose_image,
            caption=(
                "Pose points filtered "
                "to the detected dog"
            ),
            width=700
        )

        if pose_success:

            st.success(
                f"Pose detected. "
                f"Keypoints kept inside dog: "
                f"**{len(filtered_points)}**"
            )

        else:

            st.warning(
                pose_message
            )

    else:

        st.warning(
            pose_message
            if pose_message
            else "Pose image unavailable."
        )

    # ========================================================
    # THREE-WAY CONSISTENCY
    # ========================================================

    st.markdown("---")

    st.header("📊 Three-Way Consistency Analysis")

    dog_detected = st.session_state.get(
        "live_segmentation_success",
        False
    )

    pose_available = st.session_state.get(
        "live_pose_success",
        False
    )

    gradcam_available = (
        st.session_state.get(
            "live_gradcam"
        )
        is not None
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Grad-CAM",
            "Available"
            if gradcam_available
            else "Unavailable"
        )

    with c2:

        st.metric(
            "Dog Segmentation",
            "Detected"
            if dog_detected
            else "Not Detected"
        )

    with c3:

        st.metric(
            "Pose",
            "Filtered"
            if pose_available
            else "Unavailable"
        )

    if (
        gradcam_available
        and dog_detected
        and pose_available
    ):

        st.success(
            "Three-way processing is available: "
            "Grad-CAM + dog region + pose."
        )

    else:

        st.warning(
            "Three-way processing is incomplete for this image."
        )

    st.info(
        "The three-way analysis compares Grad-CAM "
        "attention with the detected dog region and "
        "pose-defined body locations."
    )

    # ========================================================
    # BODY PART XAI
    # ========================================================

    st.markdown("---")

    st.header(
        "🧠 Explainable AI — Body-Part Analysis"
    )

    cam = st.session_state.get(
        "live_cam"
    )

    if frame_id is not None:

        if (
            cam is not None
            and filtered_points
        ):

            show_live_bodypart_xai(
                cam,
                filtered_points,
                prediction,
                confidence,
                image
            )

        else:

            show_xai_explanation(
                frame_id,
                prediction,
                confidence
            )

    else:
        st.info(
            "For a new upload, body-part XAI is shown only when a compatible "
            "35-keypoint pose model is configured. The current app does not "
            "invent anatomical labels for pose outputs that lack them."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dog Emotion Recognition | "
    "Grad-CAM + YOLOv8 Segmentation + "
    "Segmentation-Filtered Full-Body Pose + "
    "Body-Part XAI"
)