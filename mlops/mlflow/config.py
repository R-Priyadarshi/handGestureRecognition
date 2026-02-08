"""
MLflow Configuration and Setup

Configures MLflow for experiment tracking and model registry.
"""

import os
from pathlib import Path
from typing import Optional


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "hand_gesture_recognition",
    artifact_location: Optional[str] = None,
):
    """
    Set up MLflow tracking.
    
    Args:
        tracking_uri: MLflow tracking URI (defaults to local ./mlruns)
        experiment_name: Name of the experiment
        artifact_location: Location to store artifacts
    """
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed. Install with: pip install mlflow")
        return None
    
    # Set tracking URI
    if tracking_uri is None:
        # Use local directory
        mlruns_dir = Path("mlruns")
        mlruns_dir.mkdir(exist_ok=True)
        tracking_uri = f"file://{mlruns_dir.absolute()}"
    
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=artifact_location,
            )
        else:
            experiment_id = experiment.experiment_id
    except Exception as e:
        print(f"Error setting up experiment: {e}")
        return None
    
    mlflow.set_experiment(experiment_name)
    
    print(f"✓ MLflow configured")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Experiment: {experiment_name}")
    
    return mlflow


def log_model_to_registry(
    model_path: str,
    model_name: str,
    version_description: str,
    metadata: Optional[dict] = None,
):
    """
    Log model to MLflow model registry.
    
    Args:
        model_path: Path to model file
        model_name: Name for the model in registry
        version_description: Description of this version
        metadata: Optional metadata dictionary
    """
    try:
        import mlflow
        from mlflow.models import infer_signature
    except ImportError:
        print("MLflow not installed.")
        return
    
    with mlflow.start_run():
        # Log model
        mlflow.pytorch.log_model(
            model_path,
            "model",
            registered_model_name=model_name,
        )
        
        # Log metadata
        if metadata:
            mlflow.log_params(metadata)
        
        # Log description
        mlflow.set_tag("description", version_description)
        
        print(f"✓ Model logged to registry: {model_name}")
