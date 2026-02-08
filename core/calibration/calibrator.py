"""
Calibration Module - Confidence Scoring and Calibration

Handles confidence calibration, threshold optimization, and explainability.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import deque


class ConfidenceCalibrator:
    """
    Calibrates confidence scores using temperature scaling or isotonic regression.
    
    Ensures that predicted confidences match actual accuracy.
    """
    
    def __init__(self, method: str = "temperature", num_classes: int = 26):
        """
        Initialize confidence calibrator.
        
        Args:
            method: Calibration method ('temperature' or 'isotonic')
            num_classes: Number of classes
        """
        self.method = method
        self.num_classes = num_classes
        self.temperature = 1.0
        self.is_calibrated = False
        
    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Calibrate on validation data.
        
        Args:
            logits: Predicted logits, shape (N, num_classes)
            labels: True labels, shape (N,)
        """
        if self.method == "temperature":
            self._calibrate_temperature(logits, labels)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented")
        
        self.is_calibrated = True
    
    def _calibrate_temperature(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        max_iter: int = 50,
    ):
        """
        Temperature scaling calibration.
        
        Args:
            logits: Predicted logits
            labels: True labels
            max_iter: Maximum optimization iterations
        """
        # Simple grid search for temperature
        best_temp = 1.0
        best_nll = float('inf')
        
        for temp in np.linspace(0.1, 5.0, 50):
            scaled_logits = logits / temp
            
            # Compute negative log likelihood
            exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            # Get probabilities of true classes
            true_probs = probs[np.arange(len(labels)), labels]
            nll = -np.mean(np.log(true_probs + 1e-10))
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
    
    def apply(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply calibration to logits.
        
        Args:
            logits: Uncalibrated logits
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_calibrated:
            # If not calibrated, just apply softmax
            scaled_logits = logits
        else:
            scaled_logits = logits / self.temperature
        
        # Apply softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        return probs


class ThresholdOptimizer:
    """
    Optimizes confidence thresholds for different metrics.
    
    Can optimize for accuracy, F1-score, or custom metrics.
    """
    
    def __init__(self, metric: str = "f1"):
        """
        Initialize threshold optimizer.
        
        Args:
            metric: Metric to optimize ('accuracy', 'f1', 'precision', 'recall')
        """
        self.metric = metric
        self.optimal_threshold = 0.5
        
    def optimize(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Find optimal confidence threshold.
        
        Args:
            confidences: Confidence scores, shape (N,)
            predictions: Predicted classes, shape (N,)
            labels: True labels, shape (N,)
            
        Returns:
            Optimal threshold
        """
        best_threshold = 0.5
        best_score = 0.0
        
        # Try different thresholds
        for threshold in np.linspace(0.1, 0.99, 50):
            # Apply threshold
            mask = confidences >= threshold
            
            if np.sum(mask) == 0:
                continue
            
            # Calculate metric
            filtered_preds = predictions[mask]
            filtered_labels = labels[mask]
            
            if self.metric == "accuracy":
                score = np.mean(filtered_preds == filtered_labels)
            elif self.metric == "f1":
                score = self._calculate_f1(filtered_preds, filtered_labels)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        return best_threshold
    
    def _calculate_f1(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Calculate macro F1 score."""
        # Simple macro F1 calculation
        classes = np.unique(labels)
        f1_scores = []
        
        for cls in classes:
            tp = np.sum((predictions == cls) & (labels == cls))
            fp = np.sum((predictions == cls) & (labels != cls))
            fn = np.sum((predictions != cls) & (labels == cls))
            
            precision = tp / (tp + fp + 1e-10)
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)
        
        return np.mean(f1_scores)


class TemporalStabilizer:
    """
    Stabilizes predictions over time using temporal smoothing.
    
    Reduces jitter in real-time gesture recognition.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        confidence_threshold: float = 0.6,
        stability_threshold: float = 0.7,
    ):
        """
        Initialize temporal stabilizer.
        
        Args:
            window_size: Number of frames to consider
            confidence_threshold: Minimum confidence for individual predictions
            stability_threshold: Fraction of frames that must agree
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.stability_threshold = stability_threshold
        
        self.prediction_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
    
    def add_prediction(
        self,
        class_id: int,
        confidence: float,
    ):
        """
        Add a new prediction to the history.
        
        Args:
            class_id: Predicted class
            confidence: Confidence score
        """
        self.prediction_history.append(class_id)
        self.confidence_history.append(confidence)
    
    def get_stable_prediction(self) -> Optional[Tuple[int, float]]:
        """
        Get stabilized prediction.
        
        Returns:
            Tuple of (class_id, confidence) if stable, None otherwise
        """
        if len(self.prediction_history) < self.window_size:
            return None
        
        # Filter by confidence threshold
        valid_predictions = [
            pred for pred, conf in zip(self.prediction_history, self.confidence_history)
            if conf >= self.confidence_threshold
        ]
        
        if not valid_predictions:
            return None
        
        # Find most common prediction
        unique, counts = np.unique(valid_predictions, return_counts=True)
        most_common_idx = np.argmax(counts)
        most_common_class = unique[most_common_idx]
        stability = counts[most_common_idx] / len(valid_predictions)
        
        # Check if stable enough
        if stability >= self.stability_threshold:
            # Average confidence of the stable prediction
            confidences = [
                conf for pred, conf in zip(self.prediction_history, self.confidence_history)
                if pred == most_common_class
            ]
            avg_confidence = np.mean(confidences)
            
            return int(most_common_class), float(avg_confidence)
        
        return None
    
    def reset(self):
        """Clear history."""
        self.prediction_history.clear()
        self.confidence_history.clear()


class ExplainabilityHooks:
    """
    Provides explainability for gesture predictions.
    
    Identifies which landmarks contribute most to predictions.
    """
    
    def __init__(self, num_landmarks: int = 21):
        """
        Initialize explainability hooks.
        
        Args:
            num_landmarks: Number of landmarks
        """
        self.num_landmarks = num_landmarks
        self.feature_importance = None
    
    def compute_feature_importance(
        self,
        model,
        landmarks: np.ndarray,
        baseline: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute feature importance using integrated gradients.
        
        Args:
            model: PyTorch model
            landmarks: Input landmarks
            baseline: Baseline for comparison (zeros if None)
            
        Returns:
            Feature importance scores
        """
        import torch
        
        if baseline is None:
            baseline = np.zeros_like(landmarks)
        
        # Convert to tensors
        landmarks_tensor = torch.from_numpy(landmarks).float().unsqueeze(0)
        baseline_tensor = torch.from_numpy(baseline).float().unsqueeze(0)
        
        landmarks_tensor.requires_grad = True
        
        # Get prediction
        model.eval()
        output = model(landmarks_tensor)
        predicted_class = torch.argmax(output, dim=-1)
        
        # Compute gradients
        model.zero_grad()
        output[0, predicted_class].backward()
        
        # Integrated gradients (simplified)
        gradients = landmarks_tensor.grad.cpu().numpy()[0]
        importance = gradients * (landmarks - baseline)
        
        self.feature_importance = importance
        return importance
    
    def get_important_landmarks(self, top_k: int = 5) -> List[int]:
        """
        Get most important landmarks.
        
        Args:
            top_k: Number of top landmarks to return
            
        Returns:
            List of landmark indices
        """
        if self.feature_importance is None:
            return []
        
        # Reshape to landmarks (assuming 2D or 3D)
        coords_per_landmark = len(self.feature_importance) // self.num_landmarks
        importance_per_landmark = []
        
        for i in range(self.num_landmarks):
            start_idx = i * coords_per_landmark
            end_idx = start_idx + coords_per_landmark
            landmark_importance = np.sum(np.abs(self.feature_importance[start_idx:end_idx]))
            importance_per_landmark.append(landmark_importance)
        
        # Get top k
        top_indices = np.argsort(importance_per_landmark)[-top_k:][::-1]
        
        return top_indices.tolist()
