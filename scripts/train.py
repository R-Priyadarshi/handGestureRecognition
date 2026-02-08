"""
Training Script

Complete training pipeline for gesture recognition models.
"""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from core.temporal.models import LightweightGestureNet, GestureTransformer
from training.datasets.gesture_dataset import GestureDataset
from training.trainers.trainer import GestureTrainer
from mlops.mlflow.config import setup_mlflow


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(config: dict):
    """Create training and validation dataloaders."""
    data_config = config['data']
    
    # Load full dataset
    full_dataset = GestureDataset(
        root_dir=data_config['root_dir'],
        use_3d=data_config.get('use_3d', False),
        cache_landmarks=True,
    )
    
    # Split into train/val
    train_size = int(len(full_dataset) * data_config.get('train_split', 0.8))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        pin_memory=True,
    )
    
    return train_loader, val_loader


def create_model(config: dict) -> nn.Module:
    """Create model based on configuration."""
    model_config = config['model']
    model_type = model_config['type']
    
    if model_type == 'lightweight':
        model = LightweightGestureNet(
            input_dim=model_config.get('input_dim', 42),
            num_classes=model_config.get('num_classes', 26),
            hidden_dims=model_config.get('hidden_dims', [128, 256, 128]),
            dropout=model_config.get('dropout', 0.2),
        )
    elif model_type == 'transformer':
        model = GestureTransformer(
            input_dim=model_config.get('input_dim', 42),
            num_classes=model_config.get('num_classes', 26),
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 4),
            dim_feedforward=model_config.get('dim_feedforward', 512),
            dropout=model_config.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train gesture recognition model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train.yaml',
        help='Path to training configuration file',
    )
    parser.add_argument(
        '--use-mlflow',
        action='store_true',
        default=True,
        help='Use MLflow for experiment tracking',
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup MLflow
    if args.use_mlflow:
        setup_mlflow(
            experiment_name=config.get('experiment_name', 'hand_gesture_recognition'),
        )
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\n📦 Loading data...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_model(config)
    print(f"  Model type: {config['model']['type']}")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")
    
    # Create optimizer
    training_config = config['training']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 0.01),
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
    )
    
    # Create trainer
    trainer = GestureTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        experiment_name=config.get('experiment_name', 'hand_gesture_recognition'),
        use_mlflow=args.use_mlflow,
        use_amp=training_config.get('use_amp', False),
        gradient_clip=training_config.get('gradient_clip', 1.0),
    )
    
    trainer.set_scheduler(scheduler)
    
    # Train
    print("\n🚀 Starting training...")
    history = trainer.train(
        num_epochs=training_config['num_epochs'],
        save_dir=config.get('output_dir', 'models'),
        early_stopping_patience=training_config.get('early_stopping_patience', 10),
    )
    
    print("\n✅ Training complete!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
