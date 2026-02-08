import argparse
import logging
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture images for hand gesture training.")
    parser.add_argument("--output-dir", default="Testing", help="Directory to store captured images.")
    parser.add_argument("--labels", type=int, default=26, help="Number of labels to capture (A-Z = 26).")
    parser.add_argument("--images-per-label", type=int, default=100, help="Images to capture per label.")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use.")
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width.")
    parser.add_argument("--height", type=int, default=640, help="Camera frame height.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera_index}.")

    camera.set(3, args.width)
    camera.set(4, args.height)

    try:
        for label in range(args.labels):
            label_dir = output_dir / str(label)
            label_dir.mkdir(parents=True, exist_ok=True)

            logging.info("Collecting images for %s", chr(label + 65))
            while True:
                success, img_from_cam = camera.read()
                if not success:
                    logging.warning("Failed to read from camera. Retrying...")
                    continue
                cv2.putText(
                    img_from_cam,
                    f'Press "Q" to collect images for {chr(label + 65)} ! :)',
                    (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (150, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Are you Ready ?", img_from_cam)
                cv2.moveWindow("Are you Ready ?", 350, 100)
                key = cv2.waitKey(25)
                if key == ord("q"):
                    break
                if key == ord("w"):
                    return

            cv2.destroyAllWindows()
            counter = 0
            while counter < args.images_per_label:
                success, img_from_cam = camera.read()
                if not success:
                    logging.warning("Failed to read from camera. Retrying...")
                    continue
                cv2.imshow(f"Capturing Images for {chr(label + 65)}", img_from_cam)
                cv2.moveWindow(f"Capturing Images for {chr(label + 65)}", 350, 100)
                cv2.waitKey(25)
                cv2.imwrite(str(label_dir / f"{counter}.jpg"), img_from_cam)
                counter += 1
            cv2.destroyAllWindows()
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
