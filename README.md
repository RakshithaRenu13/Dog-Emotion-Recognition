<<<<<<< HEAD
# Dog Emotion Classification using Deep Learning

Implementation of multiple deep learning architectures for dog emotion recognition from images. This project compares transfer learning approaches, custom CNN design, and ensemble methods for classifying dog facial expressions into five emotional states.

---

## Table of Contents

- [Overview](#overview)
- [Implemented Models](#implemented-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Prediction](#prediction)
- [Ensemble Methods](#ensemble-methods)
- [Data Augmentation](#data-augmentation)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)

---

## Overview

### Emotion Categories

The system classifies dog images into 5 emotion categories:

1. **Angry**
2. **Curious**
3. **Happy**
4. **Sad**
5. **Sleepy**

### Implemented Models

This project implements 12 different approaches:

1. **Vanilla CNN**: Custom 4-layer CNN baseline
2. **DINO + SVM**: Vision Transformer features + SVM classifier
3. **ResNet50**: Deep residual network
4. **ResNet18**: Lighter ResNet variant
5. **DenseNet121**: Densely connected network
6. **MobileNetV2**: Mobile-optimized architecture
7. **Inception v3**: Multi-scale feature extraction
8. **EfficientNet B0**: Compound scaling method
9. **VGG16**: Classic CNN baseline
10. **ConvNeXt**: Modern CNN architecture
11. **YOLO v11**: End-to-end detection and classification
12. **Ensemble**: Majority voting combination

---

## Project Structure

```
dogs/
├── training/                     # Training scripts
│   ├── train_vanilla_cnn.py     # Vanilla CNN
│   ├── train_classifier.py      # DINO + SVM
│   ├── train_cnn.py              # Transfer learning CNNs
│   └── train_yolo.py             # YOLO
│
├── prediction/                   # Prediction scripts
│   ├── predict_vanilla_cnn.py   # Vanilla CNN
│   ├── predict_emotion.py        # DINO + SVM
│   ├── predict_cnn.py            # Transfer learning CNNs
│   ├── predict_yolo.py           # YOLO
│   └── predict_ensemble.py       # Ensemble
│
├── src/                          # Core package
│   ├── config.py                 # Configuration
│   ├── models/                   # Model implementations
│   ├── utils/                    # Utilities
│   ├── visualization/            # Visualization tools
│   └── evaluation/               # Metrics
│
├── scripts/                      # Utility scripts
│   ├── generate_confusion_matrices.py
│   ├── generate_gradcam.py
│   └── augment_dataset.py
│
├── Dataset/                      # Training data
│   ├── angry/
│   ├── curious/
│   ├── happy/
│   ├── sad/
│   └── sleepy/
│
└── requirements.txt              # Dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, recommended for training)

### Setup

```bash
cd /Users/mkopc/Desktop/dogs
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
ultralytics>=8.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0
Pillow>=9.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## Quick Start

### Train a Model

```bash
# DINO + SVM (fastest)
python training/train_classifier.py

# Vanilla CNN
python training/train_vanilla_cnn.py

# Transfer learning CNNs
python training/train_cnn.py --model resnet50

# YOLO
python training/train_yolo.py
```

### Make Predictions

```bash
# DINO + SVM
python prediction/predict_emotion.py Dataset/happy/dog001.jpg

# Vanilla CNN
python prediction/predict_vanilla_cnn.py Dataset/happy/dog001.jpg

# Transfer learning CNNs
python prediction/predict_cnn.py --model resnet50 --image Dataset/happy/dog001.jpg

# YOLO
python prediction/predict_yolo.py Dataset/happy/dog001.jpg

# Ensemble
python prediction/predict_ensemble.py Dataset/happy/dog001.jpg
```

---

## Model Architectures

### 1. Vanilla CNN

Custom CNN architecture with 4 convolutional blocks:

```
Input (224×224×3)
    ↓
Conv1 (3→32) + ReLU + MaxPool
    ↓
Conv2 (32→64) + ReLU + MaxPool
    ↓
Conv3 (64→128) + ReLU + MaxPool
    ↓
Conv4 (128→256) + ReLU + MaxPool
    ↓
Flatten + Dropout(0.2) + Dense(5)
    ↓
Output (5 classes)
```

**Features**:
- 4 convolutional layers with max pooling
- ReLU activation function
- 20% dropout for regularization
- Trained from scratch (no pretrained weights)

**Usage**:
```bash
python training/train_vanilla_cnn.py --epochs 50
python prediction/predict_vanilla_cnn.py <image>
```

---

### 2. ResNet50

Deep residual network with skip connections.

**Key Feature**: Skip connections that bypass layers
```
Input → Conv Block → Output
  ↓________________↑
     Skip Connection
```

**Architecture**:
- 50 layers with residual blocks
- Bottleneck design (1×1, 3×3, 1×1 convolutions)
- 25.6M parameters

**Usage**:
```bash
python training/train_cnn.py --model resnet50
python prediction/predict_cnn.py --model resnet50 --image <path>
```

---

### 3. DenseNet121

Densely connected network where each layer connects to all subsequent layers.

**Key Feature**: Dense connections
```
Layer 1 → Layer 2 → Layer 3 → Layer 4
  ↓________↓_________↓_________↓
  └────────└─────────└─────────→ Concatenate
```

**Architecture**:
- 121 layers with dense blocks
- Feature concatenation instead of summation
- 8.0M parameters

**Usage**:
```bash
python training/train_cnn.py --model densenet121
python prediction/predict_cnn.py --model densenet121 --image <path>
```

---

### 4. MobileNetV2

Mobile-optimized architecture with depthwise separable convolutions.

**Key Feature**: Inverted residual structure
```
Narrow → Wide → Narrow (expansion then compression)
```

**Architecture**:
- Depthwise separable convolutions
- Linear bottlenecks
- 3.5M parameters

**Usage**:
```bash
python training/train_cnn.py --model mobilenet_v2
python prediction/predict_cnn.py --model mobilenet_v2 --image <path>
```

---

### 5. Inception v3

Multi-scale feature extraction with parallel convolutions.

**Key Feature**: Inception modules with multiple kernel sizes
```
Input → [1×1 Conv, 3×3 Conv, 5×5 Conv, MaxPool] → Concatenate
```

**Architecture**:
- Parallel convolutions at different scales
- Factorized convolutions
- 27.2M parameters
- Input size: 299×299

**Usage**:
```bash
python training/train_cnn.py --model inception_v3
python prediction/predict_cnn.py --model inception_v3 --image <path>
```

---

### 6. EfficientNet B0

Compound scaling method balancing depth, width, and resolution.

**Key Feature**: Balanced scaling
```
depth × width² × resolution² ≈ constant
```

**Architecture**:
- MBConv blocks with squeeze-and-excitation
- Compound scaling
- 5.3M parameters

**Usage**:
```bash
python training/train_cnn.py --model efficientnet_b0
python prediction/predict_cnn.py --model efficientnet_b0 --image <path>
```

---

### 7. ConvNeXt

Modern CNN architecture inspired by Vision Transformers.

**Key Features**:
- 7×7 convolutions (larger receptive field)
- Layer normalization
- GELU activation
- Inverted bottleneck design

**Architecture**:
- 4 hierarchical stages
- Depthwise convolutions
- 28.6M parameters (Tiny variant)

**Usage**:
```bash
python training/train_cnn.py --model convnext_tiny
python prediction/predict_cnn.py --model convnext_tiny --image <path>
```

---

### 8. YOLO v11

End-to-end object detection and classification framework.

**Architecture Components**:
- Backbone: Feature extraction
- Neck: Feature fusion
- Head: Classification

**Features**:
- Spatial attention mechanisms
- Anchor-free design
- Real-time capable

**Usage**:
```bash
python training/train_yolo.py
python prediction/predict_yolo.py <image>
```

---

### 9. DINO + SVM

Vision Transformer feature extraction with SVM classifier.

**Pipeline**:
```
Input Image → DINO (ViT) → 768-dim features → SVM → Emotion
```

**Features**:
- Self-supervised DINO features
- RBF kernel SVM
- Feature caching for efficiency

**Usage**:
```bash
python training/train_classifier.py
python prediction/predict_emotion.py <image>
```

---

### 10. Ensemble Model

Combines multiple models using majority voting.

**Default Models**: MobileNetV2, EfficientNet B0, ResNet50

**Voting Methods**:
1. **Majority Voting**: Each model votes, majority wins
2. **Average Probabilities**: Average probability distributions

**Usage**:
```bash
# Train individual models first
python training/train_cnn.py --model mobilenet_v2
python training/train_cnn.py --model efficientnet_b0
python training/train_cnn.py --model resnet50

# Use ensemble
python prediction/predict_ensemble.py <image>

# Custom models
python prediction/predict_ensemble.py --models resnet50 densenet121 inception_v3 --image <path>

# Average probabilities
python prediction/predict_ensemble.py --method average --image <path>
```

---

## Training

### Basic Training

```bash
# Vanilla CNN
python training/train_vanilla_cnn.py

# DINO + SVM
python training/train_classifier.py

# Transfer learning CNNs
python training/train_cnn.py --model resnet50
python training/train_cnn.py --model densenet121
python training/train_cnn.py --model mobilenet_v2
python training/train_cnn.py --model inception_v3
python training/train_cnn.py --model efficientnet_b0
python training/train_cnn.py --model convnext_tiny
python training/train_cnn.py --model vgg16

# YOLO
python training/train_yolo.py
```

### Training Options

```bash
python training/train_cnn.py \
    --model resnet50 \
    --epochs 25 \
    --batch-size 32 \
    --lr 0.001 \
    --val-split 0.2
```

**Parameters**:
- `--model`: Architecture choice
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--lr`: Learning rate
- `--val-split`: Validation split ratio

### Available Models for train_cnn.py

- `resnet50`, `resnet18`
- `densenet121`
- `mobilenet_v2`
- `inception_v3`
- `efficientnet_b0`, `efficientnet_b1`
- `vgg16`
- `convnext_tiny`, `convnext_small`

---

## Prediction

### Single Image

```bash
# Vanilla CNN
python prediction/predict_vanilla_cnn.py <image_path>

# DINO + SVM
python prediction/predict_emotion.py <image_path>

# Transfer learning CNNs
python prediction/predict_cnn.py --model resnet50 --image <image_path>

# YOLO
python prediction/predict_yolo.py <image_path>

# Ensemble
python prediction/predict_ensemble.py <image_path>
```

### Programmatic Usage

```python
from pathlib import Path
from src.models.cnn_classifier import CNNEmotionClassifier

# Load model
classifier = CNNEmotionClassifier(model_name='resnet50')
classifier.load(Path('models/resnet50_emotion_classifier.pth'))

# Predict
emotion, confidence, all_probs = classifier.predict('dog.jpg')
print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")
```

---

## Ensemble Methods

### Majority Voting

Each model votes for one class, majority wins.

```bash
python prediction/predict_ensemble.py --method voting --image <path>
```

### Average Probabilities

Average probability distributions from all models.

```bash
python prediction/predict_ensemble.py --method average --image <path>
```

### Custom Ensemble

```bash
python prediction/predict_ensemble.py \
    --models resnet50 densenet121 inception_v3 \
    --method average \
    --image <path>
```

---

## Data Augmentation

### Run Augmentation

```bash
python scripts/augment_dataset.py
```

**Creates**:
- `Dataset_Augmented/`: Augmented dataset (3× images)
- `Dataset_Balanced/`: Balanced class distribution

### Augmentation Techniques

- Horizontal flip
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation)
- Random crop
- Normalization (ImageNet statistics)

### Custom Augmentation

```python
from pathlib import Path
from src.utils.augmentation import augment_dataset

augment_dataset(
    input_dir=Path("Dataset"),
    output_dir=Path("Dataset_Custom"),
    num_augmentations=5,
    augmentations={
        'horizontal_flip': True,
        'rotation_range': 30,
        'color_jitter': True,
        'random_crop': True
    }
)
```

---

## Visualization

### Confusion Matrices

Generate confusion matrices for all models:

```bash
python scripts/generate_confusion_matrices.py
```

Output: `output/confusion_matrix_{model}.png`

### Grad-CAM

Generate Grad-CAM heatmaps showing model attention:

```bash
python scripts/generate_gradcam.py
```

Output: `output/failure_{n}_{true}_as_{predicted}.png`

**Grad-CAM shows**:
- Original image
- Attention heatmap
- Overlay visualization
- True vs. predicted labels

---

## Configuration

All settings are in `src/config.py`:

```python
# Paths
DATASET_PATH = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Model settings
MODEL_NAME = "facebook/dinov2-base"
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.001

# Classes
EMOTION_CLASSES = ["angry", "curious", "happy", "sad", "sleepy"]
NUM_CLASSES = 5

# Image settings
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

---

## Advanced Usage

### Transfer Learning Process

```python
# Load pretrained model
model = models.resnet50(pretrained=True)

# Freeze feature extraction layers
for param in model.parameters():
    param.requires_grad = False

# Replace classification head
model.fc = nn.Linear(2048, 5)

# Train only new head
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### Fine-Tuning

```python
from src.models.cnn_classifier import CNNEmotionClassifier

# Initialize with frozen features
classifier = CNNEmotionClassifier(model_name='resnet50', freeze_features=True)
train_loader, val_loader = classifier.prepare_data(DATASET_PATH)

# Train classification head
classifier.train(train_loader, val_loader, num_epochs=10, learning_rate=0.001)

# Unfreeze last layers
for param in classifier.model.layer4.parameters():
    param.requires_grad = True

# Fine-tune with lower learning rate
classifier.train(train_loader, val_loader, num_epochs=10, learning_rate=0.0001)
```

### Custom Dataset

1. Organize images:
```
MyDataset/
├── class1/
├── class2/
└── class3/
```

2. Update `src/config.py`:
```python
DATASET_PATH = Path("/path/to/MyDataset")
EMOTION_CLASSES = ["class1", "class2", "class3"]
NUM_CLASSES = 3
```

3. Train:
```bash
python training/train_cnn.py --model resnet50
```

### Model Export

#### ONNX Export

```python
import torch
from src.models.cnn_classifier import CNNEmotionClassifier

classifier = CNNEmotionClassifier(model_name='resnet50')
classifier.load('models/resnet50_emotion_classifier.pth')

dummy_input = torch.randn(1, 3, 224, 224).to(classifier.device)
torch.onnx.export(
    classifier.model,
    dummy_input,
    "resnet50_emotion.onnx",
    input_names=['input'],
    output_names=['output']
)
```

#### TorchScript Export

```python
scripted_model = torch.jit.trace(classifier.model, dummy_input)
scripted_model.save("resnet50_emotion.pt")
```

### Batch Prediction

```python
from pathlib import Path
from src.models.cnn_classifier import CNNEmotionClassifier

classifier = CNNEmotionClassifier(model_name='resnet50')
classifier.load(Path('models/resnet50_emotion_classifier.pth'))

image_paths = list(Path("test_images").glob("*.jpg"))

for img_path in image_paths:
    emotion, confidence, _ = classifier.predict(str(img_path))
    print(f"{img_path.name}: {emotion} ({confidence:.1%})")
```

### Ensemble with Custom Models

```python
from src.models.ensemble import EnsembleClassifier
from src.config import MODELS_DIR

# Create ensemble with custom models
ensemble = EnsembleClassifier(
    model_names=['resnet50', 'densenet121', 'inception_v3'],
    models_dir=MODELS_DIR
)

ensemble.load_models()

# Predict with majority voting
emotion, conf, details = ensemble.predict_majority_voting('dog.jpg')

# Or average probabilities
emotion, conf, probs = ensemble.predict_average_probabilities('dog.jpg')
```

---

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python training/train_cnn.py --model resnet50 --batch-size 16
```

Or use smaller model:
```bash
python training/train_cnn.py --model mobilenet_v2
```

### Model Not Found

Train the model first:
```bash
python training/train_cnn.py --model resnet50
```

### Import Errors

Install dependencies:
```bash
pip install -r requirements.txt
```

### Slow Training

Use GPU if available, or use smaller model:
```bash
python training/train_cnn.py --model mobilenet_v2
```

=======
# Dog Emotion Classification using Deep Learning

Implementation of multiple deep learning architectures for dog emotion recognition from images. This project compares transfer learning approaches, custom CNN design, and ensemble methods for classifying dog facial expressions into five emotional states.

---

## Table of Contents

- [Overview](#overview)
- [Implemented Models](#implemented-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Prediction](#prediction)
- [Ensemble Methods](#ensemble-methods)
- [Data Augmentation](#data-augmentation)
- [Visualization](#visualization)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)

---

## Overview

### Emotion Categories

The system classifies dog images into 5 emotion categories:

1. **Angry**
2. **Curious**
3. **Happy**
4. **Sad**
5. **Sleepy**

### Implemented Models

This project implements 12 different approaches:

1. **Vanilla CNN**: Custom 4-layer CNN baseline
2. **DINO + SVM**: Vision Transformer features + SVM classifier
3. **ResNet50**: Deep residual network
4. **ResNet18**: Lighter ResNet variant
5. **DenseNet121**: Densely connected network
6. **MobileNetV2**: Mobile-optimized architecture
7. **Inception v3**: Multi-scale feature extraction
8. **EfficientNet B0**: Compound scaling method
9. **VGG16**: Classic CNN baseline
10. **ConvNeXt**: Modern CNN architecture
11. **YOLO v11**: End-to-end detection and classification
12. **Ensemble**: Majority voting combination

---

## Project Structure

```
dogs/
├── training/                     # Training scripts
│   ├── train_vanilla_cnn.py     # Vanilla CNN
│   ├── train_classifier.py      # DINO + SVM
│   ├── train_cnn.py              # Transfer learning CNNs
│   └── train_yolo.py             # YOLO
│
├── prediction/                   # Prediction scripts
│   ├── predict_vanilla_cnn.py   # Vanilla CNN
│   ├── predict_emotion.py        # DINO + SVM
│   ├── predict_cnn.py            # Transfer learning CNNs
│   ├── predict_yolo.py           # YOLO
│   └── predict_ensemble.py       # Ensemble
│
├── src/                          # Core package
│   ├── config.py                 # Configuration
│   ├── models/                   # Model implementations
│   ├── utils/                    # Utilities
│   ├── visualization/            # Visualization tools
│   └── evaluation/               # Metrics
│
├── scripts/                      # Utility scripts
│   ├── generate_confusion_matrices.py
│   ├── generate_gradcam.py
│   └── augment_dataset.py
│
├── Dataset/                      # Training data
│   ├── angry/
│   ├── curious/
│   ├── happy/
│   ├── sad/
│   └── sleepy/
│
└── requirements.txt              # Dependencies
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, recommended for training)

### Setup

```bash
cd /Users/mkopc/Desktop/dogs
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
ultralytics>=8.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0
Pillow>=9.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## Quick Start

### Train a Model

```bash
# DINO + SVM (fastest)
python training/train_classifier.py

# Vanilla CNN
python training/train_vanilla_cnn.py

# Transfer learning CNNs
python training/train_cnn.py --model resnet50

# YOLO
python training/train_yolo.py
```

### Make Predictions

```bash
# DINO + SVM
python prediction/predict_emotion.py Dataset/happy/dog001.jpg

# Vanilla CNN
python prediction/predict_vanilla_cnn.py Dataset/happy/dog001.jpg

# Transfer learning CNNs
python prediction/predict_cnn.py --model resnet50 --image Dataset/happy/dog001.jpg

# YOLO
python prediction/predict_yolo.py Dataset/happy/dog001.jpg

# Ensemble
python prediction/predict_ensemble.py Dataset/happy/dog001.jpg
```

---

## Model Architectures

### 1. Vanilla CNN

Custom CNN architecture with 4 convolutional blocks:

```
Input (224×224×3)
    ↓
Conv1 (3→32) + ReLU + MaxPool
    ↓
Conv2 (32→64) + ReLU + MaxPool
    ↓
Conv3 (64→128) + ReLU + MaxPool
    ↓
Conv4 (128→256) + ReLU + MaxPool
    ↓
Flatten + Dropout(0.2) + Dense(5)
    ↓
Output (5 classes)
```

**Features**:
- 4 convolutional layers with max pooling
- ReLU activation function
- 20% dropout for regularization
- Trained from scratch (no pretrained weights)

**Usage**:
```bash
python training/train_vanilla_cnn.py --epochs 50
python prediction/predict_vanilla_cnn.py <image>
```

---

### 2. ResNet50

Deep residual network with skip connections.

**Key Feature**: Skip connections that bypass layers
```
Input → Conv Block → Output
  ↓________________↑
     Skip Connection
```

**Architecture**:
- 50 layers with residual blocks
- Bottleneck design (1×1, 3×3, 1×1 convolutions)
- 25.6M parameters

**Usage**:
```bash
python training/train_cnn.py --model resnet50
python prediction/predict_cnn.py --model resnet50 --image <path>
```

---

### 3. DenseNet121

Densely connected network where each layer connects to all subsequent layers.

**Key Feature**: Dense connections
```
Layer 1 → Layer 2 → Layer 3 → Layer 4
  ↓________↓_________↓_________↓
  └────────└─────────└─────────→ Concatenate
```

**Architecture**:
- 121 layers with dense blocks
- Feature concatenation instead of summation
- 8.0M parameters

**Usage**:
```bash
python training/train_cnn.py --model densenet121
python prediction/predict_cnn.py --model densenet121 --image <path>
```

---

### 4. MobileNetV2

Mobile-optimized architecture with depthwise separable convolutions.

**Key Feature**: Inverted residual structure
```
Narrow → Wide → Narrow (expansion then compression)
```

**Architecture**:
- Depthwise separable convolutions
- Linear bottlenecks
- 3.5M parameters

**Usage**:
```bash
python training/train_cnn.py --model mobilenet_v2
python prediction/predict_cnn.py --model mobilenet_v2 --image <path>
```

---

### 5. Inception v3

Multi-scale feature extraction with parallel convolutions.

**Key Feature**: Inception modules with multiple kernel sizes
```
Input → [1×1 Conv, 3×3 Conv, 5×5 Conv, MaxPool] → Concatenate
```

**Architecture**:
- Parallel convolutions at different scales
- Factorized convolutions
- 27.2M parameters
- Input size: 299×299

**Usage**:
```bash
python training/train_cnn.py --model inception_v3
python prediction/predict_cnn.py --model inception_v3 --image <path>
```

---

### 6. EfficientNet B0

Compound scaling method balancing depth, width, and resolution.

**Key Feature**: Balanced scaling
```
depth × width² × resolution² ≈ constant
```

**Architecture**:
- MBConv blocks with squeeze-and-excitation
- Compound scaling
- 5.3M parameters

**Usage**:
```bash
python training/train_cnn.py --model efficientnet_b0
python prediction/predict_cnn.py --model efficientnet_b0 --image <path>
```

---

### 7. ConvNeXt

Modern CNN architecture inspired by Vision Transformers.

**Key Features**:
- 7×7 convolutions (larger receptive field)
- Layer normalization
- GELU activation
- Inverted bottleneck design

**Architecture**:
- 4 hierarchical stages
- Depthwise convolutions
- 28.6M parameters (Tiny variant)

**Usage**:
```bash
python training/train_cnn.py --model convnext_tiny
python prediction/predict_cnn.py --model convnext_tiny --image <path>
```

---

### 8. YOLO v11

End-to-end object detection and classification framework.

**Architecture Components**:
- Backbone: Feature extraction
- Neck: Feature fusion
- Head: Classification

**Features**:
- Spatial attention mechanisms
- Anchor-free design
- Real-time capable

**Usage**:
```bash
python training/train_yolo.py
python prediction/predict_yolo.py <image>
```

---

### 9. DINO + SVM

Vision Transformer feature extraction with SVM classifier.

**Pipeline**:
```
Input Image → DINO (ViT) → 768-dim features → SVM → Emotion
```

**Features**:
- Self-supervised DINO features
- RBF kernel SVM
- Feature caching for efficiency

**Usage**:
```bash
python training/train_classifier.py
python prediction/predict_emotion.py <image>
```

---

### 10. Ensemble Model

Combines multiple models using majority voting.

**Default Models**: MobileNetV2, EfficientNet B0, ResNet50

**Voting Methods**:
1. **Majority Voting**: Each model votes, majority wins
2. **Average Probabilities**: Average probability distributions

**Usage**:
```bash
# Train individual models first
python training/train_cnn.py --model mobilenet_v2
python training/train_cnn.py --model efficientnet_b0
python training/train_cnn.py --model resnet50

# Use ensemble
python prediction/predict_ensemble.py <image>

# Custom models
python prediction/predict_ensemble.py --models resnet50 densenet121 inception_v3 --image <path>

# Average probabilities
python prediction/predict_ensemble.py --method average --image <path>
```

---

## Training

### Basic Training

```bash
# Vanilla CNN
python training/train_vanilla_cnn.py

# DINO + SVM
python training/train_classifier.py

# Transfer learning CNNs
python training/train_cnn.py --model resnet50
python training/train_cnn.py --model densenet121
python training/train_cnn.py --model mobilenet_v2
python training/train_cnn.py --model inception_v3
python training/train_cnn.py --model efficientnet_b0
python training/train_cnn.py --model convnext_tiny
python training/train_cnn.py --model vgg16

# YOLO
python training/train_yolo.py
```

### Training Options

```bash
python training/train_cnn.py \
    --model resnet50 \
    --epochs 25 \
    --batch-size 32 \
    --lr 0.001 \
    --val-split 0.2
```

**Parameters**:
- `--model`: Architecture choice
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size
- `--lr`: Learning rate
- `--val-split`: Validation split ratio

### Available Models for train_cnn.py

- `resnet50`, `resnet18`
- `densenet121`
- `mobilenet_v2`
- `inception_v3`
- `efficientnet_b0`, `efficientnet_b1`
- `vgg16`
- `convnext_tiny`, `convnext_small`

---

## Prediction

### Single Image

```bash
# Vanilla CNN
python prediction/predict_vanilla_cnn.py <image_path>

# DINO + SVM
python prediction/predict_emotion.py <image_path>

# Transfer learning CNNs
python prediction/predict_cnn.py --model resnet50 --image <image_path>

# YOLO
python prediction/predict_yolo.py <image_path>

# Ensemble
python prediction/predict_ensemble.py <image_path>
```

### Programmatic Usage

```python
from pathlib import Path
from src.models.cnn_classifier import CNNEmotionClassifier

# Load model
classifier = CNNEmotionClassifier(model_name='resnet50')
classifier.load(Path('models/resnet50_emotion_classifier.pth'))

# Predict
emotion, confidence, all_probs = classifier.predict('dog.jpg')
print(f"Emotion: {emotion}, Confidence: {confidence:.2%}")
```

---

## Ensemble Methods

### Majority Voting

Each model votes for one class, majority wins.

```bash
python prediction/predict_ensemble.py --method voting --image <path>
```

### Average Probabilities

Average probability distributions from all models.

```bash
python prediction/predict_ensemble.py --method average --image <path>
```

### Custom Ensemble

```bash
python prediction/predict_ensemble.py \
    --models resnet50 densenet121 inception_v3 \
    --method average \
    --image <path>
```

---

## Data Augmentation

### Run Augmentation

```bash
python scripts/augment_dataset.py
```

**Creates**:
- `Dataset_Augmented/`: Augmented dataset (3× images)
- `Dataset_Balanced/`: Balanced class distribution

### Augmentation Techniques

- Horizontal flip
- Random rotation (±20°)
- Color jitter (brightness, contrast, saturation)
- Random crop
- Normalization (ImageNet statistics)

### Custom Augmentation

```python
from pathlib import Path
from src.utils.augmentation import augment_dataset

augment_dataset(
    input_dir=Path("Dataset"),
    output_dir=Path("Dataset_Custom"),
    num_augmentations=5,
    augmentations={
        'horizontal_flip': True,
        'rotation_range': 30,
        'color_jitter': True,
        'random_crop': True
    }
)
```

---

## Visualization

### Confusion Matrices

Generate confusion matrices for all models:

```bash
python scripts/generate_confusion_matrices.py
```

Output: `output/confusion_matrix_{model}.png`

### Grad-CAM

Generate Grad-CAM heatmaps showing model attention:

```bash
python scripts/generate_gradcam.py
```

Output: `output/failure_{n}_{true}_as_{predicted}.png`

**Grad-CAM shows**:
- Original image
- Attention heatmap
- Overlay visualization
- True vs. predicted labels

---

## Configuration

All settings are in `src/config.py`:

```python
# Paths
DATASET_PATH = PROJECT_ROOT / "Dataset"
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_DIR = PROJECT_ROOT / "cache"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Model settings
MODEL_NAME = "facebook/dinov2-base"
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 0.001

# Classes
EMOTION_CLASSES = ["angry", "curious", "happy", "sad", "sleepy"]
NUM_CLASSES = 5

# Image settings
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

---

## Advanced Usage

### Transfer Learning Process

```python
# Load pretrained model
model = models.resnet50(pretrained=True)

# Freeze feature extraction layers
for param in model.parameters():
    param.requires_grad = False

# Replace classification head
model.fc = nn.Linear(2048, 5)

# Train only new head
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

### Fine-Tuning

```python
from src.models.cnn_classifier import CNNEmotionClassifier

# Initialize with frozen features
classifier = CNNEmotionClassifier(model_name='resnet50', freeze_features=True)
train_loader, val_loader = classifier.prepare_data(DATASET_PATH)

# Train classification head
classifier.train(train_loader, val_loader, num_epochs=10, learning_rate=0.001)

# Unfreeze last layers
for param in classifier.model.layer4.parameters():
    param.requires_grad = True

# Fine-tune with lower learning rate
classifier.train(train_loader, val_loader, num_epochs=10, learning_rate=0.0001)
```

### Custom Dataset

1. Organize images:
```
MyDataset/
├── class1/
├── class2/
└── class3/
```

2. Update `src/config.py`:
```python
DATASET_PATH = Path("/path/to/MyDataset")
EMOTION_CLASSES = ["class1", "class2", "class3"]
NUM_CLASSES = 3
```

3. Train:
```bash
python training/train_cnn.py --model resnet50
```

### Model Export

#### ONNX Export

```python
import torch
from src.models.cnn_classifier import CNNEmotionClassifier

classifier = CNNEmotionClassifier(model_name='resnet50')
classifier.load('models/resnet50_emotion_classifier.pth')

dummy_input = torch.randn(1, 3, 224, 224).to(classifier.device)
torch.onnx.export(
    classifier.model,
    dummy_input,
    "resnet50_emotion.onnx",
    input_names=['input'],
    output_names=['output']
)
```

#### TorchScript Export

```python
scripted_model = torch.jit.trace(classifier.model, dummy_input)
scripted_model.save("resnet50_emotion.pt")
```

### Batch Prediction

```python
from pathlib import Path
from src.models.cnn_classifier import CNNEmotionClassifier

classifier = CNNEmotionClassifier(model_name='resnet50')
classifier.load(Path('models/resnet50_emotion_classifier.pth'))

image_paths = list(Path("test_images").glob("*.jpg"))

for img_path in image_paths:
    emotion, confidence, _ = classifier.predict(str(img_path))
    print(f"{img_path.name}: {emotion} ({confidence:.1%})")
```

### Ensemble with Custom Models

```python
from src.models.ensemble import EnsembleClassifier
from src.config import MODELS_DIR

# Create ensemble with custom models
ensemble = EnsembleClassifier(
    model_names=['resnet50', 'densenet121', 'inception_v3'],
    models_dir=MODELS_DIR
)

ensemble.load_models()

# Predict with majority voting
emotion, conf, details = ensemble.predict_majority_voting('dog.jpg')

# Or average probabilities
emotion, conf, probs = ensemble.predict_average_probabilities('dog.jpg')
```

---

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python training/train_cnn.py --model resnet50 --batch-size 16
```

Or use smaller model:
```bash
python training/train_cnn.py --model mobilenet_v2
```

### Model Not Found

Train the model first:
```bash
python training/train_cnn.py --model resnet50
```

### Import Errors

Install dependencies:
```bash
pip install -r requirements.txt
```

### Slow Training

Use GPU if available, or use smaller model:
```bash
python training/train_cnn.py --model mobilenet_v2
```

>>>>>>> ae7bf5287ba48e758a0f89c1a8ec73ce04906e25
---