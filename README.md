# Hand Gesture Recognition

Industrial-grade, real-time hand gesture recognition built on open-source, free-tier tooling. The pipeline captures training images, extracts MediaPipe hand landmarks, trains a RandomForest classifier, and runs live inference from a webcam.

## Features
- Real-time hand landmark detection via MediaPipe
- Reproducible dataset creation and model training
- Configurable CLI options for camera index, paths, and confidence thresholds
- Open-source dependencies that are globally available and free to use

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Capture training images
```bash
python captureImage.py --output-dir TrainingData
```

### 2) Build dataset
```bash
python createDataset.py --input-dir TrainingData
```

### 3) Train model
```bash
python train.py
```

### 4) Run live inference
```bash
python test.py
```

## Next immediate step
Capture a small, representative training set (good lighting, consistent background, diverse hands) in `TrainingData/`, then rebuild the dataset and retrain the model:
```bash
python captureImage.py --output-dir TrainingData
python createDataset.py --input-dir TrainingData
python train.py
```
This produces a fresh `dataset.pickle` and `model.pickle` to validate live inference quality before scaling up data collection.

## Data layout
```
TrainingData/
  0/
    0.jpg
    1.jpg
  1/
    0.jpg
```
Labels are stored by numeric folder name, with `0` → `A`, `1` → `B`, etc.

## Free-tier "best in class" stack
- **MediaPipe** + **OpenCV**: fast, widely adopted, open-source computer vision.
- **scikit-learn**: proven classical ML tooling for fast iteration.
- **GitHub Actions**: zero-cost CI for automated tests.
- **Hugging Face Spaces / Streamlit Community Cloud**: free hosting for demos.

## Notes for production readiness
- Treat `dataset.pickle` and `model.pickle` as trusted artifacts (pickle is unsafe for untrusted files).
- Use separate datasets for training and evaluation to avoid data leakage.
- Consider adding monitoring and rate limits when exposing a public API.
