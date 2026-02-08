"""
Dataset Module - Gesture Dataset Loaders

Handles loading and preprocessing of gesture datasets.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import mediapipe as mp


class GestureDataset(Dataset):
    """
    PyTorch dataset for gesture recognition.
    
    Loads images, extracts landmarks, and prepares data for training.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Any] = None,
        use_3d: bool = False,
        cache_landmarks: bool = True,
    ):
        """
        Initialize gesture dataset.
        
        Args:
            root_dir: Root directory containing class subdirectories
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional transforms to apply
            use_3d: Whether to use 3D landmarks
            cache_landmarks: Whether to cache extracted landmarks
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.use_3d = use_3d
        self.cache_landmarks = cache_landmarks
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
        
        # Load dataset
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        self._load_dataset()
        
        # Cache for landmarks
        self.landmarks_cache = {} if cache_landmarks else None
    
    def _load_dataset(self):
        """Load dataset structure."""
        # Get class directories
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        # Create class mappings
        for idx, class_dir in enumerate(class_dirs):
            class_name = class_dir.name
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        # Load samples
        for class_dir in class_dirs:
            class_idx = self.class_to_idx[class_dir.name]
            
            # Get all images in class directory
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append(str(img_path))
                self.labels.append(class_idx)
            
            for img_path in class_dir.glob("*.png"):
                self.samples.append(str(img_path))
                self.labels.append(class_idx)
    
    def _extract_landmarks(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract hand landmarks from image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Landmarks array or None if hand not detected
        """
        # Check cache first
        if self.cache_landmarks and image_path in self.landmarks_cache:
            return self.landmarks_cache[image_path]
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract landmarks (first hand only)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        if self.use_3d:
            # Extract x, y, z
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            # Extract only x, y and normalize
            x_coords = []
            y_coords = []
            
            for landmark in hand_landmarks.landmark:
                x_coords.append(landmark.x)
                y_coords.append(landmark.y)
            
            # Normalize relative to bounding box
            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)
            
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.extend([
                    landmark.x - min_x,
                    landmark.y - min_y,
                ])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Cache if enabled
        if self.cache_landmarks:
            self.landmarks_cache[image_path] = landmarks
        
        return landmarks
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get dataset item.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (landmarks_tensor, label)
        """
        image_path = self.samples[idx]
        label = self.labels[idx]
        
        # Extract landmarks
        landmarks = self._extract_landmarks(image_path)
        
        # Handle missing landmarks
        if landmarks is None:
            # Return zeros
            dim = 63 if self.use_3d else 42
            landmarks = np.zeros(dim, dtype=np.float32)
        
        # Convert to tensor
        landmarks_tensor = torch.from_numpy(landmarks)
        
        # Apply transform if provided
        if self.transform is not None:
            landmarks_tensor = self.transform(landmarks_tensor)
        
        return landmarks_tensor, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get class distribution.
        
        Returns:
            Dictionary mapping class names to counts
        """
        distribution = {}
        for label in self.labels:
            class_name = self.idx_to_class[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        
        return distribution
    
    def close(self):
        """Release resources."""
        if hasattr(self, 'hands'):
            self.hands.close()


class TemporalGestureDataset(Dataset):
    """
    Dataset for temporal gesture recognition.
    
    Loads sequences of frames for dynamic gesture recognition.
    """
    
    def __init__(
        self,
        root_dir: str,
        sequence_length: int = 30,
        transform: Optional[Any] = None,
        use_3d: bool = False,
    ):
        """
        Initialize temporal gesture dataset.
        
        Args:
            root_dir: Root directory containing gesture sequences
            sequence_length: Number of frames per sequence
            transform: Optional transforms
            use_3d: Whether to use 3D landmarks
        """
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.use_3d = use_3d
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
        
        # Load sequences
        self.sequences = []
        self.labels = []
        self.class_to_idx = {}
        
        self._load_sequences()
    
    def _load_sequences(self):
        """Load sequence data."""
        # Implementation depends on dataset structure
        # This is a placeholder for sequence loading logic
        pass
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        # Placeholder for sequence loading
        pass
