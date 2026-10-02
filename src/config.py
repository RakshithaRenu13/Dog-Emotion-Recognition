"""
Configuration settings for the dog emotion classification project.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_PATH = PROJECT_ROOT / "Dataset_segmented_trainval"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configuration
MODEL_NAME = "facebook/dinov2-base"
BATCH_SIZE = 32
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Training configuration
NUM_EPOCHS = 8
LEARNING_RATE = 0.001

# Emotion classes
EMOTION_CLASSES = ["angry", "curious", "happy", "sad", "sleepy"]
NUM_CLASSES = len(EMOTION_CLASSES)

# Image configuration
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# File paths
FEATURES_CACHE = CACHE_DIR / "cached_features.npz"
SVM_MODEL_PATH = MODELS_DIR / "svm_model.joblib"

# Visualization configuration
CONFUSION_MATRIX_DPI = 150
GRADCAM_DPI = 150
COLORMAP = "Blues"

# Device configuration
import torch
DEVICE = (
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)
