"""
Landmarks Module - Normalization and Feature Extraction

Handles landmark normalization, feature extraction, and preprocessing.
"""

from typing import List, Tuple, Optional
import numpy as np


class LandmarkNormalizer:
    """
    Normalizes hand landmarks for invariance to translation, scale, and rotation.
    
    Features:
    - Translation invariance (center on wrist or palm center)
    - Scale invariance (normalize by hand size)
    - Optional rotation invariance
    - Support for 2D (x,y) and 3D (x,y,z) landmarks
    """
    
    def __init__(
        self,
        use_3d: bool = False,
        normalize_rotation: bool = False,
        reference_point: str = "wrist",  # 'wrist' or 'palm_center'
    ):
        """
        Initialize landmark normalizer.
        
        Args:
            use_3d: Whether to use 3D landmarks (x,y,z) or just 2D (x,y)
            normalize_rotation: Whether to normalize rotation
            reference_point: Point to center landmarks on ('wrist' or 'palm_center')
        """
        self.use_3d = use_3d
        self.normalize_rotation = normalize_rotation
        self.reference_point = reference_point
        
    def normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize hand landmarks.
        
        Args:
            landmarks: Array of shape (63,) for 3D or (42,) for 2D
                      Format: [x0, y0, z0, x1, y1, z1, ...] for 21 landmarks
                      
        Returns:
            Normalized landmarks with same shape
        """
        if self.use_3d:
            # Reshape to (21, 3)
            landmarks_3d = landmarks.reshape(21, 3)
            normalized = self._normalize_3d(landmarks_3d)
            return normalized.flatten()
        else:
            # Extract only x,y coordinates
            landmarks_2d = landmarks.reshape(21, 3)[:, :2]  # (21, 2)
            normalized = self._normalize_2d(landmarks_2d)
            return normalized.flatten()
    
    def _normalize_2d(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize 2D landmarks (x, y).
        
        Args:
            landmarks: Array of shape (21, 2)
            
        Returns:
            Normalized landmarks of shape (21, 2)
        """
        # Get reference point (wrist is landmark 0)
        if self.reference_point == "wrist":
            reference = landmarks[0]
        else:  # palm_center
            # Average of wrist, index MCP, pinky MCP
            reference = np.mean(landmarks[[0, 5, 17]], axis=0)
        
        # Center on reference point (translation invariance)
        centered = landmarks - reference
        
        # Normalize by hand size (scale invariance)
        # Use distance from wrist to middle finger tip as scale
        hand_size = np.linalg.norm(landmarks[12] - landmarks[0])
        if hand_size > 1e-6:  # Avoid division by zero
            normalized = centered / hand_size
        else:
            normalized = centered
        
        # Optional rotation normalization
        if self.normalize_rotation:
            normalized = self._normalize_rotation_2d(normalized)
        
        return normalized
    
    def _normalize_3d(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize 3D landmarks (x, y, z).
        
        Args:
            landmarks: Array of shape (21, 3)
            
        Returns:
            Normalized landmarks of shape (21, 3)
        """
        # Get reference point
        if self.reference_point == "wrist":
            reference = landmarks[0]
        else:  # palm_center
            reference = np.mean(landmarks[[0, 5, 17]], axis=0)
        
        # Center on reference point
        centered = landmarks - reference
        
        # Normalize by hand size (3D distance)
        hand_size = np.linalg.norm(landmarks[12] - landmarks[0])
        if hand_size > 1e-6:
            normalized = centered / hand_size
        else:
            normalized = centered
        
        # Optional rotation normalization
        if self.normalize_rotation:
            normalized = self._normalize_rotation_3d(normalized)
        
        return normalized
    
    def _normalize_rotation_2d(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize rotation in 2D by aligning wrist-to-middle-finger axis with y-axis.
        
        Args:
            landmarks: Centered landmarks of shape (21, 2)
            
        Returns:
            Rotation-normalized landmarks
        """
        # Vector from wrist to middle finger base
        vec = landmarks[9] - landmarks[0]  # Middle finger MCP
        angle = np.arctan2(vec[1], vec[0])
        
        # Rotation matrix
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        # Apply rotation
        rotated = landmarks @ rotation_matrix.T
        return rotated
    
    def _normalize_rotation_3d(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize rotation in 3D using palm plane orientation.
        
        Args:
            landmarks: Centered landmarks of shape (21, 3)
            
        Returns:
            Rotation-normalized landmarks
        """
        # Use wrist, index MCP, and pinky MCP to define palm plane
        p0 = landmarks[0]   # Wrist
        p1 = landmarks[5]   # Index MCP
        p2 = landmarks[17]  # Pinky MCP
        
        # Compute normal to palm plane
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        
        # Normalize
        normal = normal / (np.linalg.norm(normal) + 1e-6)
        
        # Create rotation matrix to align normal with z-axis
        # This is a simplified approach; for production, use Rodrigues rotation
        z_axis = np.array([0, 0, 1])
        
        # If normal is already aligned, skip rotation
        if np.abs(np.dot(normal, z_axis)) > 0.99:
            return landmarks
        
        # Rotation axis
        axis = np.cross(normal, z_axis)
        axis = axis / (np.linalg.norm(axis) + 1e-6)
        
        # Rotation angle
        angle = np.arccos(np.clip(np.dot(normal, z_axis), -1.0, 1.0))
        
        # Rodrigues rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        rotation_matrix = (
            np.eye(3) +
            np.sin(angle) * K +
            (1 - np.cos(angle)) * (K @ K)
        )
        
        # Apply rotation
        rotated = landmarks @ rotation_matrix.T
        return rotated
    
    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Extract additional features from landmarks.
        
        Features include:
        - Normalized landmark positions
        - Pairwise distances between key points
        - Finger angles
        
        Args:
            landmarks: Raw landmarks array
            
        Returns:
            Feature vector
        """
        # First normalize
        normalized = self.normalize(landmarks)
        
        # Reshape for feature extraction
        if self.use_3d:
            lm = normalized.reshape(21, 3)
        else:
            lm = normalized.reshape(21, 2)
        
        features = [normalized]  # Start with normalized landmarks
        
        # Add finger tip to wrist distances
        finger_tips = [4, 8, 12, 16, 20]  # Thumb to pinky tips
        for tip_idx in finger_tips:
            dist = np.linalg.norm(lm[tip_idx] - lm[0])
            features.append(np.array([dist]))
        
        # Add finger bend ratios (tip-to-base vs base-to-wrist)
        finger_bases = [2, 5, 9, 13, 17]  # Finger MCPs
        for tip_idx, base_idx in zip(finger_tips, finger_bases):
            tip_to_base = np.linalg.norm(lm[tip_idx] - lm[base_idx])
            base_to_wrist = np.linalg.norm(lm[base_idx] - lm[0])
            ratio = tip_to_base / (base_to_wrist + 1e-6)
            features.append(np.array([ratio]))
        
        # Concatenate all features
        return np.concatenate(features)


class TemporalFeatureExtractor:
    """
    Extracts temporal features from sequences of landmarks.
    
    Used for gesture recognition that depends on motion patterns.
    """
    
    def __init__(self, sequence_length: int = 30):
        """
        Initialize temporal feature extractor.
        
        Args:
            sequence_length: Number of frames to consider for temporal features
        """
        self.sequence_length = sequence_length
        self.history = []
        
    def add_frame(self, landmarks: np.ndarray):
        """
        Add a new frame to the temporal buffer.
        
        Args:
            landmarks: Normalized landmarks for current frame
        """
        self.history.append(landmarks)
        
        # Keep only the last N frames
        if len(self.history) > self.sequence_length:
            self.history.pop(0)
    
    def get_sequence(self) -> np.ndarray:
        """
        Get the current landmark sequence.
        
        Returns:
            Array of shape (T, D) where T is number of frames (up to sequence_length)
            and D is landmark dimension
        """
        if not self.history:
            return np.array([])
        
        return np.array(self.history)
    
    def extract_velocity_features(self) -> Optional[np.ndarray]:
        """
        Extract velocity features (temporal derivatives).
        
        Returns:
            Velocity features or None if insufficient frames
        """
        if len(self.history) < 2:
            return None
        
        # Compute frame-to-frame differences
        velocities = []
        for i in range(1, len(self.history)):
            vel = self.history[i] - self.history[i-1]
            velocities.append(vel)
        
        return np.array(velocities)
    
    def reset(self):
        """Clear the temporal buffer."""
        self.history = []
