import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the hand gesture classifier.")
    parser.add_argument("--dataset", default="dataset.pickle", help="Input dataset pickle file.")
    parser.add_argument("--model-out", default="model.pickle", help="Output model pickle file.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--estimators", type=int, default=200, help="Number of trees in the random forest."
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with dataset_path.open("rb") as file_handle:
        dataset = pickle.load(file_handle)
    data = np.asarray(dataset.get("data", []))
    labels = np.asarray(dataset.get("labels", []))

    if len(data) == 0 or len(labels) == 0:
        raise ValueError("Dataset is empty. Run createDataset.py before training.")
    if len(data) != len(labels):
        raise ValueError("Dataset samples and labels are mismatched.")

    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=args.test_size,
        shuffle=True,
        stratify=labels,
        random_state=args.random_state,
    )

    model = RandomForestClassifier(
        n_estimators=args.estimators, random_state=args.random_state
    )
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    logging.info("Accuracy: %.2f%% of samples were classified correctly.", score * 100)

    output_path = Path(args.model_out)
    with output_path.open("wb") as file_handle:
        pickle.dump({"model": model}, file_handle)
    logging.info("Saved model to %s", output_path)


if __name__ == "__main__":
    main()
