# Migration Guide: v1.x → v2.0

This guide helps you migrate from the prototype v1.x to the production-grade v2.0 platform.

## Overview of Changes

Version 2.0 is a complete architectural rewrite with the following major changes:

### ✅ What's New
- Modular, production-grade architecture
- Multi-platform inference (PyTorch, ONNX, TFLite, Web)
- Privacy-by-default design (no cloud services)
- Transformer-based temporal modeling
- MLOps integration (MLflow, DVC)
- Comprehensive testing and CI/CD
- Real-time performance optimization (<20ms latency)
- Multi-hand support
- Confidence calibration and temporal stabilization

### ⚠️ Breaking Changes
- **Removed `pickle` for model serialization** → Use ONNX/PyTorch/TFLite
- **New directory structure** → See architecture below
- **Modular API** → No more standalone scripts
- **Updated dependencies** → PyTorch 2.0+, MediaPipe 0.10+
- **New configuration system** → YAML-based configs

## Migration Steps

### 1. Update Dependencies

**Old (v1.x)**:
```txt
opencv-python
mediapipe
sklearn
numpy
```

**New (v2.0)**:
```bash
pip install -e ".[all]"
```

Or install specific components:
```bash
pip install -e ".[training,mlops]"
```

### 2. Convert Existing Models

#### Option A: Retrain with New Architecture
```bash
# Use new training script
python scripts/train.py --config configs/train.yaml
```

#### Option B: Convert Old Model (Limited Support)
```python
# If you have a sklearn model, you'll need to retrain
# sklearn models are not directly compatible with PyTorch

# 1. Extract your training data
# 2. Use new dataset loader
from training.datasets import GestureDataset

dataset = GestureDataset(root_dir="TrainingData")

# 3. Train new model
python scripts/train.py
```

### 3. Update Data Structure

**Old (v1.x)**:
```
TrainingData/
├── 0/
├── 1/
└── ...
```

**New (v2.0)** - Same structure, but with better organization:
```
TrainingData/           # Raw training data
Testing/                # Test data (optional)
data/                   # DVC-tracked datasets (optional)
  ├── raw/
  └── processed/
```

### 4. Update Code

#### Data Loading

**Old**:
```python
import pickle
dataset = pickle.load(open('dataset.pickle', 'rb'))
data = np.asarray(dataset['data'])
labels = np.asarray(dataset['labels'])
```

**New**:
```python
from training.datasets import GestureDataset
from torch.utils.data import DataLoader

dataset = GestureDataset(root_dir="TrainingData")
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### Training

**Old**:
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)

pickle.dump({'model': model}, open('model.pickle', 'wb'))
```

**New**:
```python
from core.temporal.models import LightweightGestureNet
from training.trainers import GestureTrainer

model = LightweightGestureNet(input_dim=42, num_classes=26)
trainer = GestureTrainer(model, train_loader, val_loader, ...)
history = trainer.train(num_epochs=100)
```

#### Inference

**Old**:
```python
import pickle
import mediapipe as mp

model_dict = pickle.load(open('model.pickle', 'rb'))
model = model_dict['model']

hands = mp.solutions.hands.Hands(...)
results = hands.process(image)

# Manual landmark extraction and normalization
prediction = model.predict([landmarks])
```

**New**:
```python
from core.vision import HandDetector
from core.landmarks import LandmarkNormalizer
from core.inference import InferenceEngine

detector = HandDetector()
normalizer = LandmarkNormalizer(use_3d=False)
engine = InferenceEngine(model_path="models/model.onnx")

# Clean API
detection = detector.detect(image)
normalized = normalizer.normalize(detection['landmarks'][0])
result = engine.predict(normalized)

print(f"Class: {result['class_id']}, Confidence: {result['confidence']}")
```

#### Real-Time Demo

**Old**:
```python
# test.py - monolithic script
camera = cv2.VideoCapture(0)
while True:
    # Everything in one loop
    ...
```

**New**:
```bash
# Use provided demo script
python scripts/demo.py --model models/model.onnx
```

Or use the API:
```python
from core.calibration import TemporalStabilizer

stabilizer = TemporalStabilizer(window_size=5)
# Smooth predictions over time
```

### 5. Export Models

**New Feature**: Multi-platform export

```python
from training.export import ModelExporter

exporter = ModelExporter(model, input_shape=(1, 42))

# Export to ONNX for cross-platform use
exporter.export_onnx("models/model.onnx")

# Export to TFLite for mobile
exporter.export_tflite("models/model.tflite", quantize=True)

# Or export all formats
exporter.export_all("models/exported/", model_name="gesture_model")
```

### 6. Setup MLOps (Optional but Recommended)

```bash
# Initialize MLflow
python -c "from mlops.mlflow.config import setup_mlflow; setup_mlflow()"

# Initialize DVC
python -c "from mlops.dvc.config import init_dvc; init_dvc()"

# Track training data
dvc add TrainingData
git add TrainingData.dvc .gitignore
git commit -m "Track training data with DVC"
```

## API Comparison

### Detection

| v1.x | v2.0 |
|------|------|
| Manual MediaPipe setup | `HandDetector` class |
| Manual landmark extraction | `detect()` returns structured dict |
| No multi-hand support | Multi-hand by default |

### Normalization

| v1.x | v2.0 |
|------|------|
| Manual min/max normalization | `LandmarkNormalizer` class |
| No rotation normalization | Optional rotation normalization |
| No feature extraction | Built-in feature extraction |

### Inference

| v1.x | v2.0 |
|------|------|
| sklearn RandomForest | PyTorch neural networks |
| CPU only | CPU + optional GPU |
| pickle serialization | ONNX/TFLite/PyTorch |
| No backend selection | Auto backend selection |
| No performance tracking | Built-in latency tracking |

### Confidence & Stability

| v1.x | v2.0 |
|------|------|
| Raw predictions | Calibrated confidences |
| No temporal smoothing | Temporal stabilization |
| No threshold optimization | Automatic threshold optimization |

## Configuration Files

### Training Config (v2.0)

Create `configs/train.yaml`:
```yaml
experiment_name: "hand_gesture_recognition"
output_dir: "models"

data:
  root_dir: "TrainingData"
  use_3d: false
  train_split: 0.8

model:
  type: "lightweight"
  input_dim: 42
  num_classes: 26
  hidden_dims: [128, 256, 128]
  dropout: 0.2

training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 15
```

## Testing Your Migration

1. **Verify Installation**:
```bash
python -c "import core; import training; print('✓ Import successful')"
```

2. **Run Tests**:
```bash
pytest tests/ -v
```

3. **Test Inference**:
```bash
python scripts/demo.py --model models/model.onnx
```

4. **Check Performance**:
```python
from core.inference import InferenceEngine

engine = InferenceEngine(model_path="models/model.onnx")
# Run some predictions
stats = engine.get_performance_stats()
print(f"Mean latency: {stats['mean_ms']:.2f} ms")
```

## Troubleshooting

### Issue: Cannot load old pickle models

**Solution**: Retrain using new architecture. sklearn models are not compatible with PyTorch.

### Issue: Different predictions from old model

**Solution**: This is expected. The new models are based on neural networks with better accuracy and different decision boundaries.

### Issue: Import errors

**Solution**: Ensure you've installed with the correct options:
```bash
pip install -e ".[all]"
```

### Issue: Performance slower than expected

**Solution**:
1. Use ONNX backend: `InferenceEngine(model_path="model.onnx")`
2. Use lightweight model: `model_type: "lightweight"` in config
3. Check inference stats: `engine.get_performance_stats()`

### Issue: MLflow not tracking experiments

**Solution**:
```python
from mlops.mlflow.config import setup_mlflow
setup_mlflow(experiment_name="my_experiment")
```

## Benefits of Migration

1. **Performance**: 5-10x faster inference with ONNX/TFLite
2. **Accuracy**: Better models with deep learning
3. **Portability**: Deploy to web, mobile, desktop
4. **Maintainability**: Modular, tested, documented code
5. **Reproducibility**: MLflow tracking, DVC versioning
6. **Production-Ready**: CI/CD, monitoring, error handling

## Support

If you encounter issues during migration:

1. Check this guide and README
2. Review example code in `scripts/`
3. Run the test suite to verify installation
4. Open an issue on GitHub with:
   - Your v1.x setup
   - Error messages
   - What you've tried

## Timeline

- v1.x: Prototype, proof of concept
- v2.0: Production-grade platform
- v2.1+: Planned features (see roadmap)

---

**Ready to migrate?** Start with the Quick Start in the README, then gradually update your code using this guide.
