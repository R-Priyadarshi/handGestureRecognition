"""
Unit tests for vision module.
"""

import pytest
import numpy as np
import cv2
from core.vision.hand_detector import HandDetector


@pytest.fixture
def detector():
    """Create a hand detector instance."""
    return HandDetector(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )


@pytest.fixture
def sample_image():
    """Create a simple test image."""
    # Create a blank image (BGR)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return image


def test_detector_initialization():
    """Test detector initialization."""
    detector = HandDetector()
    assert detector is not None
    assert detector.max_num_hands == 2
    assert detector.min_detection_confidence == 0.7


def test_detector_context_manager():
    """Test detector as context manager."""
    with HandDetector() as detector:
        assert detector is not None


def test_detect_no_hand(detector, sample_image):
    """Test detection with no hand in image."""
    result = detector.detect(sample_image)
    
    assert result['success'] is False
    assert result['num_hands'] == 0
    assert len(result['landmarks']) == 0
    assert len(result['handedness']) == 0


def test_detect_and_draw(detector, sample_image):
    """Test detect_and_draw method."""
    annotated, result = detector.detect_and_draw(sample_image)
    
    assert annotated.shape == sample_image.shape
    assert 'success' in result
    assert 'landmarks' in result
    assert 'handedness' in result


def test_landmark_format(detector):
    """Test landmark format when hand is detected."""
    # This would require a real image with a hand
    # For now, we just test the structure
    pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
