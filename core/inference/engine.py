"""
Inference Module - Multi-Platform Inference Engine

Supports PyTorch, ONNX, and TFLite inference with automatic fallback.
Privacy-by-default: all inference is local, no cloud services.
"""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from pathlib import Path
from enum import Enum


class InferenceBackend(Enum):
    """Supported inference backends."""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    ONNX_WEB = "onnx_web"
    TFLITE = "tflite"


class InferenceEngine:
    """
    Multi-platform inference engine with automatic backend selection.
    
    Features:
    - Automatic backend selection based on availability
    - CPU-only fallback for compatibility
    - Support for PyTorch, ONNX, and TFLite
    - Configurable confidence thresholds
    - Latency tracking for performance monitoring
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        backend: Optional[InferenceBackend] = None,
        num_classes: int = 26,
        confidence_threshold: float = 0.6,
        use_gpu: bool = False,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to model file (.pt, .onnx, or .tflite)
            backend: Inference backend to use (auto-detect if None)
            num_classes: Number of gesture classes
            confidence_threshold: Minimum confidence for predictions
            use_gpu: Whether to use GPU if available (ONNX only)
        """
        self.model_path = Path(model_path)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        
        # Detect backend if not specified
        if backend is None:
            backend = self._detect_backend()
        
        self.backend = backend
        self.model = None
        self.session = None
        
        # Load model
        self._load_model()
        
        # Performance tracking
        self.inference_times = []
    
    def _detect_backend(self) -> InferenceBackend:
        """Auto-detect best available backend."""
        suffix = self.model_path.suffix.lower()
        
        if suffix == ".onnx":
            return InferenceBackend.ONNX
        elif suffix == ".tflite":
            return InferenceBackend.TFLITE
        elif suffix in [".pt", ".pth"]:
            return InferenceBackend.PYTORCH
        else:
            raise ValueError(f"Unknown model format: {suffix}")
    
    def _load_model(self):
        """Load model based on backend."""
        if self.backend == InferenceBackend.PYTORCH:
            self._load_pytorch()
        elif self.backend == InferenceBackend.ONNX:
            self._load_onnx()
        elif self.backend == InferenceBackend.TFLITE:
            self._load_tflite()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _load_pytorch(self):
        """Load PyTorch model."""
        import torch
        
        self.model = torch.load(self.model_path, map_location='cpu')
        self.model.eval()
        
        # Move to GPU if requested and available
        if self.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def _load_onnx(self):
        """Load ONNX model."""
        import onnxruntime as ort
        
        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Set execution providers
        providers = ['CPUExecutionProvider']
        if self.use_gpu:
            providers.insert(0, 'CUDAExecutionProvider')
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def _load_tflite(self):
        """Load TFLite model."""
        try:
            import tensorflow as tf
            
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        except ImportError:
            raise ImportError("TensorFlow Lite not installed. Install with: pip install tensorflow")
    
    def predict(
        self,
        landmarks: np.ndarray,
        return_all_scores: bool = False,
    ) -> Dict[str, Any]:
        """
        Predict gesture class from landmarks.
        
        Args:
            landmarks: Normalized landmark array
            return_all_scores: Whether to return scores for all classes
            
        Returns:
            Dictionary containing:
            - class_id: Predicted class ID
            - confidence: Confidence score
            - all_scores: Scores for all classes (if return_all_scores=True)
            - backend: Inference backend used
            - inference_time_ms: Inference time in milliseconds
        """
        import time
        
        start_time = time.perf_counter()
        
        # Run inference based on backend
        if self.backend == InferenceBackend.PYTORCH:
            result = self._predict_pytorch(landmarks)
        elif self.backend == InferenceBackend.ONNX:
            result = self._predict_onnx(landmarks)
        elif self.backend == InferenceBackend.TFLITE:
            result = self._predict_tflite(landmarks)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
        
        # Calculate inference time
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        
        # Extract class and confidence
        class_id = int(np.argmax(result))
        confidence = float(result[class_id])
        
        # Build result dictionary
        output = {
            "class_id": class_id,
            "confidence": confidence,
            "backend": self.backend.value,
            "inference_time_ms": inference_time,
            "meets_threshold": confidence >= self.confidence_threshold,
        }
        
        if return_all_scores:
            output["all_scores"] = result.tolist()
        
        return output
    
    def _predict_pytorch(self, landmarks: np.ndarray) -> np.ndarray:
        """PyTorch inference."""
        import torch
        
        # Convert to tensor
        x = torch.from_numpy(landmarks).float().unsqueeze(0)
        
        if self.use_gpu and torch.cuda.is_available():
            x = x.cuda()
        
        # Inference
        with torch.no_grad():
            if hasattr(self.model, 'predict'):
                _, probs = self.model.predict(x)
                probs = probs.cpu().numpy()[0]
            else:
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        return probs
    
    def _predict_onnx(self, landmarks: np.ndarray) -> np.ndarray:
        """ONNX inference."""
        # Prepare input
        x = landmarks.astype(np.float32).reshape(1, -1)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )
        
        # Apply softmax if output is logits
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        return probs
    
    def _predict_tflite(self, landmarks: np.ndarray) -> np.ndarray:
        """TFLite inference."""
        # Prepare input
        x = landmarks.astype(np.float32).reshape(1, -1)
        
        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'], x)
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Apply softmax if needed
        exp_output = np.exp(output - np.max(output))
        probs = exp_output / np.sum(exp_output)
        
        return probs
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with mean, min, max, p95 inference times
        """
        if not self.inference_times:
            return {}
        
        times = np.array(self.inference_times)
        
        return {
            "mean_ms": float(np.mean(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "num_inferences": len(times),
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times = []


class BatchInferenceEngine:
    """
    Batch inference engine for processing multiple samples efficiently.
    """
    
    def __init__(self, engine: InferenceEngine, batch_size: int = 32):
        """
        Initialize batch inference engine.
        
        Args:
            engine: Base inference engine
            batch_size: Batch size for inference
        """
        self.engine = engine
        self.batch_size = batch_size
    
    def predict_batch(
        self,
        landmarks_batch: np.ndarray,
    ) -> list:
        """
        Predict for a batch of landmarks.
        
        Args:
            landmarks_batch: Array of shape (N, D) where N is batch size
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        # Process in batches
        for i in range(0, len(landmarks_batch), self.batch_size):
            batch = landmarks_batch[i:i + self.batch_size]
            
            for landmarks in batch:
                result = self.engine.predict(landmarks)
                results.append(result)
        
        return results
