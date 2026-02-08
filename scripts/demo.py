"""
Inference Demo Script

Real-time gesture recognition demo using webcam.
"""

import argparse
import cv2
import numpy as np
import torch

from core.vision.hand_detector import HandDetector
from core.landmarks.normalizer import LandmarkNormalizer
from core.calibration.calibrator import TemporalStabilizer
from core.inference.engine import InferenceEngine, InferenceBackend


# Gesture labels (A-Z)
GESTURE_LABELS = {i: chr(65 + i) for i in range(26)}


def main():
    parser = argparse.ArgumentParser(description='Real-time gesture recognition demo')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model file (.pt, .onnx, or .tflite)',
    )
    parser.add_argument(
        '--backend',
        type=str,
        choices=['pytorch', 'onnx', 'tflite'],
        default=None,
        help='Inference backend (auto-detect if not specified)',
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.6,
        help='Minimum confidence threshold',
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Camera device ID',
    )
    
    args = parser.parse_args()
    
    # Initialize components
    print("🚀 Initializing gesture recognition system...")
    
    # Hand detector
    detector = HandDetector(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    
    # Landmark normalizer
    normalizer = LandmarkNormalizer(
        use_3d=False,
        normalize_rotation=False,
        reference_point="wrist",
    )
    
    # Inference engine
    backend_map = {
        'pytorch': InferenceBackend.PYTORCH,
        'onnx': InferenceBackend.ONNX,
        'tflite': InferenceBackend.TFLITE,
    }
    
    backend = backend_map.get(args.backend) if args.backend else None
    
    engine = InferenceEngine(
        model_path=args.model,
        backend=backend,
        num_classes=26,
        confidence_threshold=args.confidence_threshold,
        use_gpu=False,  # CPU-only for compatibility
    )
    
    # Temporal stabilizer
    stabilizer = TemporalStabilizer(
        window_size=5,
        confidence_threshold=args.confidence_threshold,
        stability_threshold=0.7,
    )
    
    print(f"✓ Using backend: {engine.backend.value}")
    print(f"✓ Confidence threshold: {args.confidence_threshold}")
    
    # Open camera
    camera = cv2.VideoCapture(args.camera_id)
    
    if not camera.isOpened():
        print(f"Error: Could not open camera {args.camera_id}")
        return
    
    print("\n🎥 Starting camera feed...")
    print("Press 'q' to quit\n")
    
    # Main loop
    frame_count = 0
    
    while True:
        ret, frame = camera.read()
        
        if not ret:
            print("Error reading frame")
            break
        
        frame_count += 1
        
        # Detect hands
        annotated_frame, detection = detector.detect_and_draw(frame)
        
        if detection['success']:
            # Get landmarks (first hand only)
            landmarks = detection['landmarks'][0]
            
            # Normalize landmarks
            normalized = normalizer.normalize(landmarks)
            
            # Run inference
            result = engine.predict(normalized, return_all_scores=False)
            
            # Add to stabilizer
            stabilizer.add_prediction(
                result['class_id'],
                result['confidence'],
            )
            
            # Get stable prediction
            stable_pred = stabilizer.get_stable_prediction()
            
            if stable_pred is not None:
                class_id, confidence = stable_pred
                gesture = GESTURE_LABELS.get(class_id, "?")
                
                # Display prediction
                text = f"{gesture} ({confidence:.2f})"
                color = (0, 255, 0) if confidence >= args.confidence_threshold else (0, 165, 255)
                
                cv2.putText(
                    annotated_frame,
                    text,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    color,
                    3,
                    cv2.LINE_AA,
                )
            
            # Display inference time
            if result['inference_time_ms'] < 20:
                fps_color = (0, 255, 0)  # Green for good performance
            else:
                fps_color = (0, 165, 255)  # Orange for slow
            
            cv2.putText(
                annotated_frame,
                f"{result['inference_time_ms']:.1f} ms",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                fps_color,
                2,
                cv2.LINE_AA,
            )
        else:
            # No hand detected
            cv2.putText(
                annotated_frame,
                "No hand detected",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        
        # Display frame
        cv2.imshow('Hand Gesture Recognition', annotated_frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    detector.close()
    
    # Print performance stats
    stats = engine.get_performance_stats()
    if stats:
        print("\n📊 Performance Statistics:")
        print(f"  Mean inference time: {stats['mean_ms']:.2f} ms")
        print(f"  P95 inference time: {stats['p95_ms']:.2f} ms")
        print(f"  P99 inference time: {stats['p99_ms']:.2f} ms")
        print(f"  Total inferences: {stats['num_inferences']}")


if __name__ == '__main__':
    main()
