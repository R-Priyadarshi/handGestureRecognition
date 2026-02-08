import argparse
import logging
import pickle
from pathlib import Path

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset from labeled training images.")
    parser.add_argument("--input-dir", default="TrainingData", help="Directory containing labeled images.")
    parser.add_argument("--output", default="dataset.pickle", help="Output dataset pickle file.")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.3,
        help="Minimum hand detection confidence.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    root_directory = Path(args.input_dir)
    if not root_directory.exists():
        raise FileNotFoundError(f"Training data directory not found: {root_directory}")

    data = []
    labels = []

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True, min_detection_confidence=args.min_detection_confidence
    ) as hands:
        for label_dir in sorted(root_directory.iterdir()):
            if not label_dir.is_dir():
                continue
            for img_path in sorted(label_dir.iterdir()):
                if not img_path.is_file():
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    logging.warning("Skipping unreadable image %s", img_path)
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)
                if not results.multi_hand_landmarks:
                    logging.warning("No hand landmarks detected for %s", img_path)
                    continue

                for hand_landmarks in results.multi_hand_landmarks:
                    rel_pos_landmarks = []
                    x_coord_landmarks = []
                    y_coord_landmarks = []
                    for landmark in hand_landmarks.landmark:
                        x_coord_landmarks.append(landmark.x)
                        y_coord_landmarks.append(landmark.y)

                    min_x = min(x_coord_landmarks)
                    min_y = min(y_coord_landmarks)
                    for landmark in hand_landmarks.landmark:
                        rel_pos_landmarks.append(landmark.x - min_x)
                        rel_pos_landmarks.append(landmark.y - min_y)

                    if rel_pos_landmarks:
                        data.append(rel_pos_landmarks)
                        labels.append(label_dir.name)

    if not data:
        raise ValueError("No hand landmark samples were created. Check the training data.")

    logging.info("Dataset samples: %s", len(data))
    output_path = Path(args.output)
    with output_path.open("wb") as file_handle:
        pickle.dump({"data": data, "labels": labels}, file_handle)
    logging.info("Saved dataset to %s", output_path)


if __name__ == "__main__":
    main()
