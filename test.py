import argparse
import logging
import pickle
import string
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time hand gesture inference.")
    parser.add_argument("--model", default="model.pickle", help="Path to the trained model.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use.")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.3,
        help="Minimum hand detection confidence.",
    )
    parser.add_argument("--max-hands", type=int, default=1, help="Maximum hands to track.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    with model_path.open("rb") as file_handle:
        model_dict = pickle.load(file_handle)
    model = model_dict.get("model")
    if model is None:
        raise ValueError("Model file is missing the trained classifier.")

    labels_dict = {index: label for index, label in enumerate(string.ascii_uppercase)}

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}.")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=args.min_detection_confidence,
            max_num_hands=args.max_hands,
        ) as hands:
            while True:
                success, img_from_cam = camera.read()
                if not success:
                    logging.warning("Failed to read from camera. Retrying...")
                    continue

                height, width, _ = img_from_cam.shape
                img_rgb = cv2.cvtColor(img_from_cam, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                predicted_character = "Unknown"
                if results.multi_hand_landmarks:
                    x_coord_landmarks = []
                    y_coord_landmarks = []
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            img_from_cam,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style(),
                        )
                        for landmark in hand_landmarks.landmark:
                            x_coord_landmarks.append(landmark.x)
                            y_coord_landmarks.append(landmark.y)

                    landmarks = []
                    if x_coord_landmarks and y_coord_landmarks:
                        min_x = min(x_coord_landmarks)
                        min_y = min(y_coord_landmarks)
                        for hand_landmarks in results.multi_hand_landmarks:
                            for landmark in hand_landmarks.landmark:
                                landmarks.append(landmark.x - min_x)
                                landmarks.append(landmark.y - min_y)

                    if landmarks:
                        prediction = model.predict([np.asarray(landmarks)])
                        predicted_character = labels_dict.get(int(prediction[0]), "Unknown")

                        margin = 10
                        x1 = max(int(min_x * width) - margin, 0)
                        y1 = max(int(min_y * height) - margin, 0)
                        x2 = min(int(max(x_coord_landmarks) * width) + margin, width - 1)
                        y2 = min(int(max(y_coord_landmarks) * height) + margin, height - 1)

                        cv2.rectangle(img_from_cam, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(
                            img_from_cam,
                            predicted_character,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.3,
                            (0, 0, 0),
                            3,
                            cv2.LINE_AA,
                        )

                cv2.imshow("frame", img_from_cam)
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
