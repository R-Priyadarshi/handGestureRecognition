"""
Vision Module - MediaPipe Hands Integration

Handles hand detection and landmark extraction using MediaPipe Hands.
Privacy-by-default: all processing is local, no cloud inference.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import mediapipe as mp
import cv2


class HandDetector:
    """
    Hand detector using MediaPipe Hands.
    
    Features:
    - Multi-hand support (up to max_num_hands)
    - Configurable confidence thresholds
    - Static image or video mode
    - Returns normalized landmarks and handedness
    """
    
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ):
        """
        Initialize hand detector.
        
        Args:
            static_image_mode: If True, treats each image as independent
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for landmark tracking
            model_complexity: Model complexity (0=lite, 1=full)
        """
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
        
    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect hands in an image.
        
        Args:
            image: Input image in BGR format (OpenCV format)
            
        Returns:
            Dictionary containing:
            - landmarks: List of hand landmarks (21 points per hand, x,y,z)
            - handedness: List of handedness labels ('Left' or 'Right')
            - success: Whether detection was successful
            - num_hands: Number of hands detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return {
                "landmarks": [],
                "handedness": [],
                "success": False,
                "num_hands": 0,
            }
        
        # Extract landmarks and handedness
        landmarks_list = []
        handedness_list = []
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            # Extract 21 landmarks (x, y, z)
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks_list.append(np.array(landmarks))
            handedness_list.append(handedness.classification[0].label)
        
        return {
            "landmarks": landmarks_list,
            "handedness": handedness_list,
            "success": True,
            "num_hands": len(landmarks_list),
        }
    
    def detect_and_draw(
        self,
        image: np.ndarray,
        draw_landmarks: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Detect hands and optionally draw landmarks on image.
        
        Args:
            image: Input image in BGR format
            draw_landmarks: Whether to draw landmarks on the image
            
        Returns:
            Tuple of (annotated_image, detection_results)
        """
        # Make a copy to avoid modifying original
        annotated_image = image.copy()
        
        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return annotated_image, {
                "landmarks": [],
                "handedness": [],
                "success": False,
                "num_hands": 0,
            }
        
        # Draw landmarks if requested
        if draw_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        
        # Extract landmarks and handedness
        landmarks_list = []
        handedness_list = []
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks_list.append(np.array(landmarks))
            handedness_list.append(handedness.classification[0].label)
        
        detection_results = {
            "landmarks": landmarks_list,
            "handedness": handedness_list,
            "success": True,
            "num_hands": len(landmarks_list),
        }
        
        return annotated_image, detection_results
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
