"""
Temporal Module - Transformer-based Temporal Modeling

Implements transformer architecture for temporal gesture recognition.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class GestureTransformer(nn.Module):
    """
    Transformer-based temporal gesture recognition model.
    
    Architecture:
    - Input embedding layer
    - Positional encoding
    - Transformer encoder layers
    - Classification head
    
    Supports both single-frame and temporal sequence inputs.
    """
    
    def __init__(
        self,
        input_dim: int = 42,  # 21 landmarks * 2 (x, y)
        num_classes: int = 26,  # A-Z gestures
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 60,
    ):
        """
        Initialize gesture transformer.
        
        Args:
            input_dim: Dimension of input landmarks
            num_classes: Number of gesture classes
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim) or (batch, input_dim)
            src_mask: Optional mask for padding
            
        Returns:
            Tuple of (logits, features):
            - logits: Class logits of shape (batch, num_classes)
            - features: Encoded features of shape (batch, d_model)
        """
        # Handle single-frame input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        batch_size, seq_len, _ = x.shape
        
        # Input embedding: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        x = self.input_embedding(x)
        
        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x, src_mask)
        
        # Use the last timestep for classification (or mean pooling)
        # Here we use mean pooling across sequence
        features = encoded.mean(dim=0)  # (batch, d_model)
        
        # Classification
        logits = self.classifier(features)
        
        return logits, features
    
    def predict(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict gesture class.
        
        Args:
            x: Input tensor
            temperature: Temperature for softmax (lower = more confident)
            
        Returns:
            Tuple of (predicted_class, confidence_scores)
        """
        logits, _ = self.forward(x)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Softmax to get probabilities
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Get predicted class
        predicted_class = torch.argmax(probs, dim=-1)
        
        return predicted_class, probs


class LightweightGestureNet(nn.Module):
    """
    Lightweight CNN-based gesture recognition model.
    
    For real-time performance on CPU and mobile devices.
    Designed for single-frame static gesture recognition.
    """
    
    def __init__(
        self,
        input_dim: int = 42,
        num_classes: int = 26,
        hidden_dims: list = [128, 256, 128],
        dropout: float = 0.2,
    ):
        """
        Initialize lightweight gesture net.
        
        Args:
            input_dim: Dimension of input landmarks
            num_classes: Number of gesture classes
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict gesture class.
        
        Args:
            x: Input tensor
            temperature: Temperature for softmax
            
        Returns:
            Tuple of (predicted_class, confidence_scores)
        """
        logits = self.forward(x)
        
        # Apply temperature scaling
        scaled_logits = logits / temperature
        
        # Softmax to get probabilities
        probs = torch.softmax(scaled_logits, dim=-1)
        
        # Get predicted class
        predicted_class = torch.argmax(probs, dim=-1)
        
        return predicted_class, probs
