import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TRAINING_DIR = REPO_ROOT / "TrainingData"
DATASET_PATH = REPO_ROOT / "dataset.pickle"
MODEL_PATH = REPO_ROOT / "model.pickle"


def has_any_files(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for _root, _dirs, files in os.walk(directory):
        if files:
            return True
    return False


if not has_any_files(TRAINING_DIR):
    next_step = (
        "Collect labeled training images in TrainingData "
        "(e.g., by using captureImage.py and ensuring it writes there)."
    )
elif not DATASET_PATH.exists():
    next_step = "Run createDataset.py to generate dataset.pickle from TrainingData."
elif not MODEL_PATH.exists():
    next_step = "Run train.py to train and save model.pickle."
else:
    next_step = "Run test.py to start live hand-gesture recognition."

print(f"What to do next:\n{next_step}")
