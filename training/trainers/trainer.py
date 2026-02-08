"""
Trainer Module - PyTorch Training Loop with MLflow Integration

Production-grade training pipeline with:
- MLflow experiment tracking
- Mixed precision training
- Learning rate scheduling
- Early stopping
- Model checkpointing
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class GestureTrainer:
    """
    Production-grade trainer for gesture recognition models.
    
    Features:
    - MLflow experiment tracking
    - Automatic mixed precision (AMP)
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    - Gradient clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        experiment_name: str = "gesture_recognition",
        use_mlflow: bool = True,
        use_amp: bool = False,
        gradient_clip: Optional[float] = 1.0,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            experiment_name: MLflow experiment name
            use_mlflow: Whether to use MLflow tracking
            use_amp: Whether to use automatic mixed precision
            gradient_clip: Max gradient norm for clipping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow
        self.use_amp = use_amp
        self.gradient_clip = gradient_clip
        
        # Initialize MLflow if requested
        if self.use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                self.mlflow.set_experiment(experiment_name)
            except ImportError:
                print("MLflow not installed. Tracking disabled.")
                self.use_mlflow = False
        
        # AMP scaler for mixed precision
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Scheduler (optional)
        self.scheduler = None
    
    def set_scheduler(self, scheduler):
        """Set learning rate scheduler."""
        self.scheduler = scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.gradient_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return {
            'train_loss': avg_loss,
            'train_acc': accuracy,
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store for per-class metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        # Calculate per-class accuracy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        unique_classes = np.unique(all_labels)
        class_accuracies = {}
        
        for cls in unique_classes:
            mask = all_labels == cls
            if np.sum(mask) > 0:
                class_acc = np.mean(all_predictions[mask] == all_labels[mask])
                class_accuracies[int(cls)] = float(class_acc)
        
        return {
            'val_loss': avg_loss,
            'val_acc': accuracy,
            'class_accuracies': class_accuracies,
        }
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = "models",
        early_stopping_patience: int = 10,
    ) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
        }
        
        patience_counter = 0
        
        # Start MLflow run
        if self.use_mlflow:
            with self.mlflow.start_run():
                # Log hyperparameters
                self.mlflow.log_params({
                    'num_epochs': num_epochs,
                    'optimizer': type(self.optimizer).__name__,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'use_amp': self.use_amp,
                })
                
                return self._training_loop(
                    num_epochs,
                    save_path,
                    early_stopping_patience,
                    history,
                )
        else:
            return self._training_loop(
                num_epochs,
                save_path,
                early_stopping_patience,
                history,
            )
    
    def _training_loop(
        self,
        num_epochs: int,
        save_path: Path,
        early_stopping_patience: int,
        history: Dict[str, list],
    ) -> Dict[str, Any]:
        """Internal training loop."""
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['train_acc'].append(train_metrics['train_acc'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_acc'])
            
            # Log to MLflow
            if self.use_mlflow:
                self.mlflow.log_metrics({
                    'train_loss': train_metrics['train_loss'],
                    'train_acc': train_metrics['train_acc'],
                    'val_loss': val_metrics['val_loss'],
                    'val_acc': val_metrics['val_acc'],
                }, step=epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.2f}%")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_acc']:.2f}%")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step(val_metrics['val_loss'])
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"Learning rate: {current_lr}")
            
            # Save best model
            if val_metrics['val_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['val_acc']
                self.best_val_loss = val_metrics['val_loss']
                
                checkpoint_path = save_path / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_metrics['val_acc'],
                    'val_loss': val_metrics['val_loss'],
                }, checkpoint_path)
                
                print(f"Saved best model with val_acc: {val_metrics['val_acc']:.2f}%")
                
                # Reset patience
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        final_checkpoint_path = save_path / "final_model.pt"
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, final_checkpoint_path)
        
        return history
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
