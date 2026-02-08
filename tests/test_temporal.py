"""
Unit tests for temporal models.
"""

import pytest
import torch
from core.temporal.models import LightweightGestureNet, GestureTransformer


@pytest.fixture
def input_2d():
    """Create sample 2D input."""
    return torch.randn(4, 42)  # Batch of 4, 42 features


@pytest.fixture
def input_sequence():
    """Create sample sequence input."""
    return torch.randn(4, 10, 42)  # Batch of 4, 10 frames, 42 features


def test_lightweight_model_creation():
    """Test LightweightGestureNet creation."""
    model = LightweightGestureNet(
        input_dim=42,
        num_classes=26,
        hidden_dims=[128, 256, 128],
    )
    
    assert model is not None
    assert model.input_dim == 42
    assert model.num_classes == 26


def test_lightweight_forward(input_2d):
    """Test forward pass of LightweightGestureNet."""
    model = LightweightGestureNet(input_dim=42, num_classes=26)
    
    output = model(input_2d)
    
    assert output.shape == (4, 26)  # Batch size 4, 26 classes


def test_lightweight_predict(input_2d):
    """Test predict method."""
    model = LightweightGestureNet(input_dim=42, num_classes=26)
    model.eval()
    
    with torch.no_grad():
        predicted_class, probs = model.predict(input_2d)
    
    assert predicted_class.shape == (4,)
    assert probs.shape == (4, 26)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


def test_transformer_model_creation():
    """Test GestureTransformer creation."""
    model = GestureTransformer(
        input_dim=42,
        num_classes=26,
        d_model=128,
        nhead=8,
        num_layers=2,
    )
    
    assert model is not None
    assert model.input_dim == 42
    assert model.num_classes == 26
    assert model.d_model == 128


def test_transformer_forward_single_frame(input_2d):
    """Test transformer with single frame input."""
    model = GestureTransformer(
        input_dim=42,
        num_classes=26,
        d_model=128,
        nhead=8,
        num_layers=2,
    )
    
    logits, features = model(input_2d)
    
    assert logits.shape == (4, 26)
    assert features.shape == (4, 128)


def test_transformer_forward_sequence(input_sequence):
    """Test transformer with sequence input."""
    model = GestureTransformer(
        input_dim=42,
        num_classes=26,
        d_model=128,
        nhead=8,
        num_layers=2,
    )
    
    logits, features = model(input_sequence)
    
    assert logits.shape == (4, 26)
    assert features.shape == (4, 128)


def test_transformer_predict(input_sequence):
    """Test transformer predict method."""
    model = GestureTransformer(input_dim=42, num_classes=26)
    model.eval()
    
    with torch.no_grad():
        predicted_class, probs = model.predict(input_sequence)
    
    assert predicted_class.shape == (4,)
    assert probs.shape == (4, 26)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)


def test_model_training_mode():
    """Test that models can be set to training mode."""
    model = LightweightGestureNet()
    
    model.train()
    assert model.training is True
    
    model.eval()
    assert model.training is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
