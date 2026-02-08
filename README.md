# Hand Gesture Recognition

Hand sign recognition project that extracts MediaPipe hand landmarks and trains a RandomForest classifier to predict A-Z letters from a webcam feed.

## Features
- Image capture workflow for building a labeled dataset.
- MediaPipe landmark extraction for consistent feature vectors.
- RandomForest training pipeline and real-time inference script.

## Requirements
- Python 3.9+ (3.10 recommended)
- Webcam (for capture and live inference)
- Packages: `opencv-python`, `mediapipe`, `scikit-learn`, `numpy`

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python mediapipe scikit-learn numpy
# Optional: pin versions for reproducible runs
pip freeze > requirements.txt
```

For deployments, install with `pip install -r requirements.txt` to keep training and inference aligned.

## Data Collection
`captureImage.py` collects sample images per letter.

```bash
python captureImage.py
```

`captureImage.py` writes captures to `TrainingData/` by default. This matches the input expected by `createDataset.py`.

If you already have images under `Testing/`, move/rename that folder to `TrainingData/` before building the dataset.

## Build the Dataset
`createDataset.py` scans `TrainingData/<label>/` folders and writes `dataset.pickle`.

```bash
python createDataset.py
```

## Train the Model
`train.py` trains a RandomForest and writes `model.pickle`.

```bash
python train.py
```

## Run Live Inference
`test.py` loads `model.pickle` and streams predictions from your webcam.

```bash
python test.py
```

Press `q` to exit the window.

## Configuration Tips
- Adjust the number of images per letter in `captureImage.py` to tune dataset size.
- Use consistent lighting/backgrounds across captures to improve accuracy.
- Keep a dedicated validation split (e.g., 80/20 as in `train.py`) or a separate validation folder for use during training.

## Production-Grade Guidance
- **Reproducibility:** Pin dependency versions (e.g., `pip freeze > requirements.txt`) and store dataset metadata (sample counts, capture date, camera settings) in JSON/YAML or a data versioning tool like DVC.
- **Model versioning:** Store `model.pickle` with a version tag and track accuracy metrics alongside it.
- **Data governance:** Obtain consent for any captured imagery; keep raw frames out of production logs.
- **Runtime hardening:** Wrap inference in a service (e.g., FastAPI) and add health checks, timeouts, and input validation.
- **Performance:** Use lower-resolution capture when targeting CPU-only environments and consider batching landmark extraction for offline inference.

## Free-Tier Service Recommendations
| Use case | Service options | Notes |
| :--- | :--- | :--- |
| GPU/CPU training | Google Colab / Kaggle Notebooks | Free GPUs with session limits; great for model experiments. |
| Demo UI hosting | Hugging Face Spaces (Gradio/Streamlit) | Rapid public demos with community-friendly hosting. |
| Lightweight web apps | Streamlit Community Cloud | Ideal for simple dashboards and live demos. |
| API hosting | Render / Fly.io free tier | Suitable for low-traffic inference APIs with sleep modes. |
| CI for scripts | GitHub Actions | Automate linting/tests once added. |

## Repository Layout
- `captureImage.py`: collect labeled images.
- `createDataset.py`: extract landmark features and build `dataset.pickle`.
- `train.py`: train and export `model.pickle`.
- `test.py`: real-time prediction from webcam.
- `TrainingData/`: default capture output directory used for dataset creation.
- `Testing/`: legacy capture folder retained in the repo for backward compatibility (rename to `TrainingData/` only if you already have data there).

## Limitations
- The current pipeline assumes a single hand in frame and letters A-Z.
- Real-time inference requires a working webcam and adequate lighting.
