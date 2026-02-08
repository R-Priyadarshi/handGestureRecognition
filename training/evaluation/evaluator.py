"""
Evaluation Module - Model Evaluation and Metrics

Comprehensive evaluation metrics for gesture recognition models.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelEvaluator:
    """
    Comprehensive model evaluator.
    
    Features:
    - Accuracy, precision, recall, F1 metrics
    - Per-class metrics
    - Confusion matrix
    - Calibration metrics
    - Performance benchmarks
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model to evaluate
            device: Device to run evaluation on
            class_names: Optional list of class names
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        
        # Evaluation results
        self.predictions = []
        self.ground_truth = []
        self.confidences = []
        self.inference_times = []
    
    def evaluate(
        self,
        data_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: DataLoader for evaluation data
            verbose: Whether to print progress
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        self.predictions = []
        self.ground_truth = []
        self.confidences = []
        self.inference_times = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Time inference
                import time
                start_time = time.perf_counter()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate inference time
                inference_time = (time.perf_counter() - start_time) * 1000  # ms
                self.inference_times.append(inference_time)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=-1)
                confidences, predicted = probs.max(1)
                
                # Store results
                self.predictions.extend(predicted.cpu().numpy())
                self.ground_truth.extend(labels.cpu().numpy())
                self.confidences.extend(confidences.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive metrics."""
        predictions = np.array(self.predictions)
        ground_truth = np.array(self.ground_truth)
        confidences = np.array(self.confidences)
        
        # Overall metrics
        accuracy = accuracy_score(ground_truth, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth,
            predictions,
            average=None,
            zero_division=0,
        )
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            ground_truth,
            predictions,
            average='macro',
            zero_division=0,
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            ground_truth,
            predictions,
            average='weighted',
            zero_division=0,
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions)
        
        # Performance metrics
        inference_times = np.array(self.inference_times)
        
        # Build metrics dictionary
        metrics = {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1),
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist(),
            'confusion_matrix': cm.tolist(),
            'mean_confidence': float(np.mean(confidences)),
            'mean_inference_time_ms': float(np.mean(inference_times)),
            'p95_inference_time_ms': float(np.percentile(inference_times, 95)),
            'p99_inference_time_ms': float(np.percentile(inference_times, 99)),
        }
        
        return metrics
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in readable format."""
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        
        print(f"\nPrecision/Recall/F1:")
        print(f"  Macro Precision: {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall: {metrics['macro_recall']:.4f}")
        print(f"  Weighted Precision: {metrics['weighted_precision']:.4f}")
        print(f"  Weighted Recall: {metrics['weighted_recall']:.4f}")
        
        print(f"\nPerformance:")
        print(f"  Mean Inference Time: {metrics['mean_inference_time_ms']:.2f} ms")
        print(f"  P95 Inference Time: {metrics['p95_inference_time_ms']:.2f} ms")
        print(f"  P99 Inference Time: {metrics['p99_inference_time_ms']:.2f} ms")
        print(f"  Mean Confidence: {metrics['mean_confidence']:.4f}")
        
        print("\n" + "=" * 60)
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Optional path to save plot
            normalize: Whether to normalize the matrix
        """
        cm = np.array(confusion_matrix(self.ground_truth, self.predictions))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_per_class_metrics(
        self,
        save_path: Optional[str] = None,
    ):
        """
        Plot per-class precision, recall, and F1.
        
        Args:
            save_path: Optional path to save plot
        """
        precision = np.array(self.predictions)
        recall = np.array(self.ground_truth)
        
        precision, recall, f1, support = precision_recall_fscore_support(
            self.ground_truth,
            self.predictions,
            average=None,
            zero_division=0,
        )
        
        x = np.arange(len(precision))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        
        if self.class_names:
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        else:
            ax.set_xticklabels([str(i) for i in x])
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def save_report(
        self,
        output_dir: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Save comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save report
            metrics: Optional pre-computed metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if metrics is None:
            metrics = self._calculate_metrics()
        
        # Save metrics as JSON
        import json
        with open(output_path / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save text report
        with open(output_path / "report.txt", "w") as f:
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Macro F1-Score: {metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted F1-Score: {metrics['weighted_f1']:.4f}\n\n")
            
            f.write("Performance:\n")
            f.write(f"  Mean Inference Time: {metrics['mean_inference_time_ms']:.2f} ms\n")
            f.write(f"  P95 Inference Time: {metrics['p95_inference_time_ms']:.2f} ms\n")
            f.write(f"  P99 Inference Time: {metrics['p99_inference_time_ms']:.2f} ms\n\n")
            
            # Per-class metrics
            f.write("\nPer-Class Metrics:\n")
            f.write("-" * 60 + "\n")
            for i, (p, r, f, s) in enumerate(zip(
                metrics['per_class_precision'],
                metrics['per_class_recall'],
                metrics['per_class_f1'],
                metrics['per_class_support'],
            )):
                class_name = self.class_names[i] if self.class_names else str(i)
                f.write(f"{class_name}: P={p:.3f}, R={r:.3f}, F1={f:.3f}, Support={s}\n")
        
        # Plot and save visualizations
        self.plot_confusion_matrix(save_path=str(output_path / "confusion_matrix.png"))
        self.plot_per_class_metrics(save_path=str(output_path / "per_class_metrics.png"))
        
        print(f"\nEvaluation report saved to: {output_path}")
