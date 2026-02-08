#!/usr/bin/env python3
"""
Setup and Verification Script

Verifies the installation and environment setup.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"  ✗ Python {version.major}.{version.minor} detected")
        print(f"  ✗ Python 3.9+ required")
        return False
    
    print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'numpy',
        'opencv-python',
        'mediapipe',
        'torch',
        'onnx',
        'onnxruntime',
        'yaml',
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').split('[')[0])
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (not installed)")
            missing.append(package)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Install with: pip install {' '.join(missing)}")
        return False
    
    return True


def check_directory_structure():
    """Check if directory structure is correct."""
    print("\n📁 Checking directory structure...")
    
    required_dirs = [
        'core',
        'core/vision',
        'core/landmarks',
        'core/temporal',
        'core/inference',
        'core/calibration',
        'training',
        'training/datasets',
        'training/trainers',
        'training/evaluation',
        'training/export',
        'apps',
        'mlops',
        'scripts',
        'configs',
        'tests',
        'docs',
    ]
    
    base_path = Path('.')
    all_exist = True
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            all_exist = False
    
    return all_exist


def check_imports():
    """Check if core modules can be imported."""
    print("\n🔍 Checking core imports...")
    
    try:
        from core.vision import HandDetector
        print("  ✓ core.vision.HandDetector")
    except ImportError as e:
        print(f"  ✗ core.vision.HandDetector: {e}")
        return False
    
    try:
        from core.landmarks import LandmarkNormalizer
        print("  ✓ core.landmarks.LandmarkNormalizer")
    except ImportError as e:
        print(f"  ✗ core.landmarks.LandmarkNormalizer: {e}")
        return False
    
    try:
        from core.temporal.models import LightweightGestureNet
        print("  ✓ core.temporal.models.LightweightGestureNet")
    except ImportError as e:
        print(f"  ✗ core.temporal.models.LightweightGestureNet: {e}")
        return False
    
    try:
        from core.inference import InferenceEngine
        print("  ✓ core.inference.InferenceEngine")
    except ImportError as e:
        print(f"  ✗ core.inference.InferenceEngine: {e}")
        return False
    
    return True


def check_gpu():
    """Check GPU availability."""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    CUDA Version: {torch.version.cuda}")
        else:
            print(f"  ℹ CUDA not available (CPU mode)")
    except Exception as e:
        print(f"  ⚠ Could not check GPU: {e}")


def run_quick_test():
    """Run a quick functionality test."""
    print("\n🧪 Running quick test...")
    
    try:
        import numpy as np
        from core.vision import HandDetector
        from core.landmarks import LandmarkNormalizer
        
        # Create detector
        detector = HandDetector()
        print("  ✓ HandDetector initialized")
        
        # Create normalizer
        normalizer = LandmarkNormalizer(use_3d=False)
        print("  ✓ LandmarkNormalizer initialized")
        
        # Test normalization
        landmarks = np.random.randn(63).astype(np.float32)
        normalized = normalizer.normalize(landmarks)
        assert normalized.shape == (42,)
        print("  ✓ Normalization works")
        
        return True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("Hand Gesture Recognition Platform - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version()),
        ("Dependencies", check_dependencies()),
        ("Directory Structure", check_directory_structure()),
        ("Core Imports", check_imports()),
    ]
    
    # Optional checks
    check_gpu()
    
    # Quick test
    test_passed = run_quick_test()
    checks.append(("Quick Test", test_passed))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(result for _, result in checks)
    
    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All checks passed! You're ready to go.")
        print("\nNext steps:")
        print("  1. Collect training data: python captureImage.py")
        print("  2. Train a model: python scripts/train.py")
        print("  3. Run demo: python scripts/demo.py --model models/model.onnx")
        print("\nSee README.md for detailed instructions.")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -e '.[all]'")
        print("  - Check Python version (3.9+ required)")
        print("  - Ensure you're in the correct directory")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
