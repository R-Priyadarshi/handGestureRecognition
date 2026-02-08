"""
DVC Configuration

Sets up DVC for data version control.
"""

import subprocess
from pathlib import Path
from typing import Optional


def init_dvc(
    remote_name: str = "storage",
    remote_url: Optional[str] = None,
):
    """
    Initialize DVC for data versioning.
    
    Args:
        remote_name: Name of the remote storage
        remote_url: URL of the remote storage (e.g., gdrive://folder_id)
    """
    try:
        # Check if DVC is installed
        subprocess.run(["dvc", "version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("DVC not installed. Install with: pip install dvc")
        return False
    
    # Check if already initialized
    if Path(".dvc").exists():
        print("DVC already initialized")
        return True
    
    # Initialize DVC
    try:
        subprocess.run(["dvc", "init"], check=True)
        print("✓ DVC initialized")
        
        # Add remote if provided
        if remote_url:
            subprocess.run([
                "dvc", "remote", "add", "-d", remote_name, remote_url
            ], check=True)
            print(f"✓ DVC remote added: {remote_name}")
        
        # Commit DVC config
        subprocess.run(["git", "add", ".dvc", ".dvcignore"], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error initializing DVC: {e}")
        return False


def track_data(
    data_path: str,
    message: Optional[str] = None,
):
    """
    Track data with DVC.
    
    Args:
        data_path: Path to data directory or file
        message: Optional commit message
    """
    try:
        # Add data to DVC
        subprocess.run(["dvc", "add", data_path], check=True)
        print(f"✓ Added {data_path} to DVC")
        
        # Add .dvc file to git
        dvc_file = f"{data_path}.dvc"
        subprocess.run(["git", "add", dvc_file, ".gitignore"], check=True)
        
        # Commit
        if message:
            subprocess.run(["git", "commit", "-m", message], check=True)
            print(f"✓ Committed DVC tracking")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error tracking data: {e}")
        return False


def create_dvc_pipeline():
    """Create DVC pipeline for reproducible training."""
    pipeline_config = """
stages:
  prepare_data:
    cmd: python scripts/prepare_data.py
    deps:
      - scripts/prepare_data.py
      - data/raw
    outs:
      - data/processed

  train_model:
    cmd: python scripts/train.py
    deps:
      - scripts/train.py
      - data/processed
      - core/
      - training/
    params:
      - train.yaml:
          - model
          - training
    outs:
      - models/trained_model.pt
    metrics:
      - metrics/train_metrics.json:
          cache: false

  evaluate_model:
    cmd: python scripts/evaluate.py
    deps:
      - scripts/evaluate.py
      - models/trained_model.pt
      - data/processed
    metrics:
      - metrics/eval_metrics.json:
          cache: false

  export_model:
    cmd: python scripts/export.py
    deps:
      - scripts/export.py
      - models/trained_model.pt
    outs:
      - models/exported/model.onnx
      - models/exported/model.tflite
"""
    
    with open("dvc.yaml", "w") as f:
        f.write(pipeline_config)
    
    print("✓ DVC pipeline created: dvc.yaml")
