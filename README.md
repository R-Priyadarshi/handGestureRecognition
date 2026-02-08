# Hand Gesture Recognition Platform

[![CI](https://github.com/R-Priyadarshi/handGestureRecognition/actions/workflows/ci.yml/badge.svg)](https://github.com/R-Priyadarshi/handGestureRecognition/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, multi-platform hand-gesture intelligence platform with privacy-by-default design, real-time performance, and industrial-grade architecture.

## 🚀 Features

### Core Capabilities
- **Multi-Platform Inference**: PyTorch, ONNX, TFLite, and ONNX Runtime Web/WebGPU support
- **Privacy-First**: All processing is local—no cloud inference, no image/video egress
- **Real-Time Performance**: <20ms latency target, 60 FPS capable, CPU-optimized
- **Production-Ready**: Modular architecture, comprehensive testing, CI/CD integration

### Technical Highlights
- **MediaPipe Hands Integration**: Robust hand landmark detection
- **Transformer-Based Temporal Modeling**: Advanced gesture recognition
- **Multi-Hand Support**: Detect and track multiple hands simultaneously
- **Confidence Calibration**: Temperature scaling and threshold optimization
- **Temporal Stabilization**: Smooth predictions over time
- **Explainability Hooks**: Understand which landmarks matter most

### MLOps Integration
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data version control and reproducible pipelines
- **Automated Testing**: Unit tests, integration tests, property-based tests
- **CI/CD**: GitHub Actions for automated testing and deployment

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Training](#training)
- [Inference](#inference)
- [Model Export](#model-export)
- [Web Deployment](#web-deployment)
- [API Documentation](#api-documentation)
- [Migration Guide](#migration-guide)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/R-Priyadarshi/handGestureRecognition.git
cd handGestureRecognition

# Install base dependencies
pip install -e .
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Full Installation (All Features)

```bash
# Install all optional dependencies
pip install -e ".[all]"
```

### Installation Options

- `training`: MLflow, TensorBoard, evaluation tools
- `mlops`: DVC and data versioning tools
- `export`: ONNX and TFLite export tools
- `dev`: Testing, linting, and development tools
- `web`: Web application dependencies
- `all`: Everything above

## 🏃 Quick Start

### Real-Time Demo

```bash
# Run real-time gesture recognition with webcam
python scripts/demo.py --model models/gesture_model.onnx --confidence-threshold 0.6
```

### Training a Model

```bash
# Train with default configuration
python scripts/train.py --config configs/train.yaml

# Train with MLflow tracking
python scripts/train.py --config configs/train.yaml --use-mlflow
```

### Exporting Models

```bash
# Export to all formats (ONNX, TFLite, quantized TFLite)
python scripts/export.py --model models/best_model.pt --output-dir models/exported
```

## 🏗️ Architecture

```
hand-gesture-platform/
├── core/                    # Core ML modules
│   ├── vision/             # MediaPipe Hands integration
│   ├── landmarks/          # Landmark normalization and features
│   ├── temporal/           # Transformer-based temporal models
│   ├── inference/          # Multi-backend inference engine
│   └── calibration/        # Confidence calibration and explainability
│
├── training/               # Training pipeline
│   ├── datasets/          # Dataset loaders
│   ├── trainers/          # Training loops with MLflow
│   ├── evaluation/        # Model evaluation and metrics
│   └── export/            # Model export (ONNX, TFLite)
│
├── apps/                   # Application interfaces
│   ├── web/               # Web app (ONNX Runtime Web)
│   ├── mobile/            # Mobile app structure
│   └── desktop/           # Desktop app structure
│
├── mlops/                  # MLOps tooling
│   ├── mlflow/            # MLflow configuration
│   ├── dvc/               # DVC pipelines
│   └── ci/                # CI/CD scripts
│
├── scripts/                # Executable scripts
│   ├── train.py           # Training script
│   ├── demo.py            # Real-time demo
│   └── export.py          # Model export script
│
├── configs/                # Configuration files
│   └── train.yaml         # Training configuration
│
├── tests/                  # Test suite
│   ├── test_vision.py
│   ├── test_landmarks.py
│   └── test_temporal.py
│
└── docs/                   # Documentation
    ├── API.md
    ├── MIGRATION.md
    └── DEPLOYMENT.md
```

### Module Descriptions

#### `core/vision`
- **HandDetector**: MediaPipe Hands wrapper with multi-hand support
- Real-time hand detection and landmark extraction
- Configurable confidence thresholds

#### `core/landmarks`
- **LandmarkNormalizer**: Translation, scale, and rotation normalization
- **TemporalFeatureExtractor**: Temporal sequence management
- Feature extraction for ML models

#### `core/temporal`
- **GestureTransformer**: Transformer-based temporal model
- **LightweightGestureNet**: CNN-based model for real-time inference
- Positional encoding and attention mechanisms

#### `core/inference`
- **InferenceEngine**: Multi-backend inference (PyTorch, ONNX, TFLite)
- Automatic backend selection and CPU fallback
- Performance tracking and benchmarking

#### `core/calibration`
- **ConfidenceCalibrator**: Temperature scaling
- **ThresholdOptimizer**: Optimize confidence thresholds
- **TemporalStabilizer**: Smooth predictions over time
- **ExplainabilityHooks**: Feature importance analysis

## 🎯 Training

### Prepare Data

Organize your dataset:
```
TrainingData/
├── 0/          # Class 0 (e.g., 'A')
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── 1/          # Class 1 (e.g., 'B')
│   └── ...
└── ...
```

### Configure Training

Edit `configs/train.yaml`:
```yaml
data:
  root_dir: "TrainingData"
  use_3d: false
  train_split: 0.8

model:
  type: "lightweight"  # or "transformer"
  input_dim: 42
  num_classes: 26

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### Run Training

```bash
# Basic training
python scripts/train.py

# With MLflow tracking
python scripts/train.py --use-mlflow

# Custom config
python scripts/train.py --config my_config.yaml
```

### Monitor Training

```bash
# View MLflow UI
mlflow ui

# Then open http://localhost:5000
```

## 🔮 Inference

### Python API

```python
from core.vision import HandDetector
from core.landmarks import LandmarkNormalizer
from core.inference import InferenceEngine
import cv2

# Initialize components
detector = HandDetector(max_num_hands=1)
normalizer = LandmarkNormalizer(use_3d=False)
engine = InferenceEngine(
    model_path="models/gesture_model.onnx",
    confidence_threshold=0.6
)

# Process frame
frame = cv2.imread("hand_image.jpg")
detection = detector.detect(frame)

if detection['success']:
    landmarks = detection['landmarks'][0]
    normalized = normalizer.normalize(landmarks)
    result = engine.predict(normalized)
    
    print(f"Gesture: {result['class_id']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Inference time: {result['inference_time_ms']:.1f} ms")
```

### Real-Time Inference

```python
from core.calibration import TemporalStabilizer

stabilizer = TemporalStabilizer(window_size=5)

while True:
    # Detect and predict (as above)
    stabilizer.add_prediction(result['class_id'], result['confidence'])
    
    # Get stable prediction
    stable = stabilizer.get_stable_prediction()
    if stable:
        class_id, confidence = stable
        print(f"Stable prediction: {class_id} ({confidence:.2f})")
```

## 📦 Model Export

### Export to ONNX

```python
from training.export import ModelExporter
import torch

model = torch.load("models/best_model.pt")
exporter = ModelExporter(
    model=model,
    input_shape=(1, 42),
)

exporter.export_onnx(
    output_path="models/gesture_model.onnx",
    opset_version=14,
    simplify=True,
    verify=True,
)
```

### Export to TFLite

```python
exporter.export_tflite(
    output_path="models/gesture_model.tflite",
    quantize=True,
    quantization_mode="dynamic",
)
```

### Export All Formats

```bash
python scripts/export.py \
    --model models/best_model.pt \
    --output-dir models/exported
```

## 🌐 Web Deployment

### ONNX Runtime Web

```javascript
// Load model
const session = await ort.InferenceSession.create('gesture_model.onnx', {
    executionProviders: ['webgpu', 'wasm']
});

// Run inference
const feeds = { 'input': new ort.Tensor('float32', landmarks, [1, 42]) };
const results = await session.run(feeds);
const predictions = results.output.data;
```

See `apps/web/` for complete web application.

## 📊 Performance Targets

| Metric | Target | Typical |
|--------|--------|---------|
| Inference Latency (CPU) | < 20ms | 8-15ms |
| Frame Rate | 60 FPS | 60+ FPS |
| Model Size (ONNX) | < 5 MB | 2-4 MB |
| Model Size (TFLite quantized) | < 1 MB | 0.5-1 MB |
| Accuracy (A-Z gestures) | > 95% | 96-98% |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=core --cov=training

# Run specific test file
pytest tests/test_vision.py -v

# Run property-based tests
pytest tests/test_landmarks.py -v
```

## 🔄 Migration from v1.x

See [MIGRATION.md](docs/MIGRATION.md) for detailed migration guide.

### Breaking Changes
- Removed unsafe `pickle` serialization
- Replaced script-only workflow with modular API
- New directory structure
- Updated dependencies (PyTorch 2.0+, MediaPipe 0.10+)

### Quick Migration

**Old (v1.x)**:
```python
import pickle
model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']
```

**New (v2.0)**:
```python
from core.inference import InferenceEngine
engine = InferenceEngine(model_path="models/model.onnx")
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md).

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black core training scripts
isort core training scripts
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe**: Hand landmark detection
- **PyTorch**: Deep learning framework
- **ONNX**: Model interoperability
- **MLflow**: Experiment tracking

## 📮 Contact

For questions and support, please open an issue on GitHub.

---

**Note**: This is a major rewrite (v2.0) with breaking changes. See [MIGRATION.md](docs/MIGRATION.md) for upgrade instructions.
