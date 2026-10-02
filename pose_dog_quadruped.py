from pathlib import Path
import deeplabcut

PROJECT_ROOT = Path(r"C:\Users\acer\dogs-emotion-recognition")

POSE_VIDEO = PROJECT_ROOT / "dog_pose_outputs" / "dog_test_images.mp4"
POSE_OUTPUT_DIR = PROJECT_ROOT / "dog_pose_outputs" / "results"

print("=" * 70)
print("DOG POSE ESTIMATION - DEEPLABCUT SUPERANIMAL QUADRUPED")
print("=" * 70)

print("Video exists :", POSE_VIDEO.exists())
print("Video size   :", POSE_VIDEO.stat().st_size, "bytes")
print("Video        :", POSE_VIDEO)
print("Output       :", POSE_OUTPUT_DIR)
print("=" * 70)

if not POSE_VIDEO.exists():
    raise FileNotFoundError(f"Video not found: {POSE_VIDEO}")

POSE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

deeplabcut.video_inference_superanimal(
    videos=str(POSE_VIDEO),
    superanimal_name="superanimal_quadruped",
    model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2",
    max_individuals=1,
    dest_folder=str(POSE_OUTPUT_DIR),
)

print("\n" + "=" * 70)
print("POSE ESTIMATION COMPLETED")
print("=" * 70)

print("Results saved to:")
print(POSE_OUTPUT_DIR)