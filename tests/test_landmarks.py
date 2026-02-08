"""
Unit tests for landmarks module.
"""

import pytest
import numpy as np
from core.landmarks.normalizer import LandmarkNormalizer, TemporalFeatureExtractor
from hypothesis import given, strategies as st


@pytest.fixture
def normalizer_2d():
    """Create a 2D landmark normalizer."""
    return LandmarkNormalizer(use_3d=False, normalize_rotation=False)


@pytest.fixture
def normalizer_3d():
    """Create a 3D landmark normalizer."""
    return LandmarkNormalizer(use_3d=True, normalize_rotation=False)


@pytest.fixture
def sample_landmarks_2d():
    """Create sample 2D landmarks."""
    # 21 landmarks with x, y, z coordinates
    landmarks = np.random.randn(63).astype(np.float32)
    return landmarks


@pytest.fixture
def sample_landmarks_3d():
    """Create sample 3D landmarks."""
    landmarks = np.random.randn(63).astype(np.float32)
    return landmarks


def test_normalizer_initialization():
    """Test normalizer initialization."""
    normalizer = LandmarkNormalizer()
    assert normalizer.use_3d is False
    assert normalizer.normalize_rotation is False
    assert normalizer.reference_point == "wrist"


def test_normalize_2d(normalizer_2d, sample_landmarks_2d):
    """Test 2D normalization."""
    normalized = normalizer_2d.normalize(sample_landmarks_2d)
    
    # Check output shape (21 landmarks * 2 coordinates)
    assert normalized.shape == (42,)
    assert normalized.dtype == np.float32


def test_normalize_3d(normalizer_3d, sample_landmarks_3d):
    """Test 3D normalization."""
    normalized = normalizer_3d.normalize(sample_landmarks_3d)
    
    # Check output shape (21 landmarks * 3 coordinates)
    assert normalized.shape == (63,)
    assert normalized.dtype == np.float32


def test_extract_features(normalizer_2d, sample_landmarks_2d):
    """Test feature extraction."""
    features = normalizer_2d.extract_features(sample_landmarks_2d)
    
    # Should include normalized landmarks + additional features
    assert len(features) > 42


@given(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False))
def test_normalization_scale_invariance(scale):
    """Test that normalization is scale invariant."""
    normalizer = LandmarkNormalizer(use_3d=False)
    
    # Create landmarks
    landmarks = np.random.randn(63).astype(np.float32)
    
    # Scale landmarks
    scaled_landmarks = landmarks * scale if scale != 0 else landmarks
    
    # Normalize both
    norm1 = normalizer.normalize(landmarks)
    norm2 = normalizer.normalize(scaled_landmarks)
    
    # They should be similar (allowing for numerical precision)
    if scale != 0:
        assert np.allclose(norm1, norm2, atol=1e-3)


def test_temporal_feature_extractor():
    """Test temporal feature extractor."""
    extractor = TemporalFeatureExtractor(sequence_length=10)
    
    # Add frames
    for i in range(15):
        landmarks = np.random.randn(42).astype(np.float32)
        extractor.add_frame(landmarks)
    
    # Get sequence (should only keep last 10)
    sequence = extractor.get_sequence()
    assert sequence.shape == (10, 42)
    
    # Get velocity features
    velocities = extractor.extract_velocity_features()
    assert velocities is not None
    assert velocities.shape == (9, 42)  # 10 frames = 9 velocities


def test_temporal_reset():
    """Test temporal buffer reset."""
    extractor = TemporalFeatureExtractor()
    
    # Add some frames
    for i in range(5):
        extractor.add_frame(np.random.randn(42).astype(np.float32))
    
    # Reset
    extractor.reset()
    
    # Should be empty
    sequence = extractor.get_sequence()
    assert len(sequence) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
