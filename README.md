# GestureSphere

GestureSphere is a hand gesture recognition system that uses computer vision and machine learning to recognize American Sign Language (ASL) alphabet gestures in real-time.

## Features

- Real-time hand gesture detection and recognition
- Support for all 26 letters of the ASL alphabet (A-Z)
- Uses MediaPipe for hand landmark detection
- Random Forest classifier for gesture classification
- Interactive image capture interface for training data collection

## Project Components

- **test.py**: Main application for real-time gesture recognition via webcam
- **captureImage.py**: Utility to capture training images for each gesture
- **createDataset.py**: Processes captured images and creates the training dataset
- **train.py**: Trains the Random Forest model on the dataset

## Requirements

- Python 3.x
- OpenCV (cv2)
- MediaPipe
- scikit-learn
- NumPy
- pickle

## Installation

```bash
pip install opencv-python mediapipe scikit-learn numpy
```

## Usage

### 1. Capture Training Images

Run the image capture utility to collect training data for each letter:

```bash
python captureImage.py
```

Follow the on-screen instructions to capture images for each gesture.

### 2. Create Dataset

Process the captured images to extract hand landmarks:

```bash
python createDataset.py
```

This will create a `dataset.pickle` file containing the processed training data.

### 3. Train the Model

Train the Random Forest classifier:

```bash
python train.py
```

This will create a `model.pickle` file containing the trained model.

### 4. Run Real-time Recognition

Launch the GestureSphere application:

```bash
python test.py
```

Show hand gestures to the camera to see real-time recognition. Press 'q' to quit.

## How It Works

1. **Hand Detection**: MediaPipe detects hand landmarks in each frame
2. **Feature Extraction**: Relative positions of hand landmarks are computed
3. **Classification**: Random Forest model predicts the gesture based on landmark positions
4. **Display**: The recognized letter is displayed on the video feed

## License

This project is open source and available for educational and research purposes.

## Project Name

GestureSphere - Spherical understanding of hand gestures in 3D space.
