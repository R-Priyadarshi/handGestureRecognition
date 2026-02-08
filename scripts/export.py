"""
Model Export Script

Export trained PyTorch models to ONNX, TFLite, and quantized formats.
"""

import argparse
from pathlib import Path
import torch

from core.temporal.models import LightweightGestureNet, GestureTransformer
from training.export.exporter import ModelExporter


def load_model(model_path: str, model_type: str = "lightweight", input_dim: int = 42, num_classes: int = 26):
    """Load trained model."""
    # Create model
    if model_type == "lightweight":
        model = LightweightGestureNet(
            input_dim=input_dim,
            num_classes=num_classes,
        )
    elif model_type == "transformer":
        model = GestureTransformer(
            input_dim=input_dim,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # Checkpoint is the model itself
        model = checkpoint
    
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Export gesture recognition model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pt file)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/exported',
        help='Directory to save exported models',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='gesture_model',
        help='Base name for exported models',
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='lightweight',
        choices=['lightweight', 'transformer'],
        help='Type of model architecture',
    )
    parser.add_argument(
        '--input-dim',
        type=int,
        default=42,
        help='Input dimension (21 landmarks * 2 for 2D)',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=26,
        help='Number of gesture classes',
    )
    parser.add_argument(
        '--formats',
        nargs='+',
        default=['onnx', 'tflite'],
        choices=['onnx', 'tflite', 'tflite_quant'],
        help='Formats to export',
    )
    
    args = parser.parse_args()
    
    # Load model
    print(f"\n📦 Loading model from {args.model}...")
    model = load_model(
        args.model,
        model_type=args.model_type,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
    )
    print("✓ Model loaded successfully")
    
    # Create exporter
    exporter = ModelExporter(
        model=model,
        input_shape=(1, args.input_dim),
        device='cpu',
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Export to requested formats
    if 'onnx' in args.formats:
        print("\n📦 Exporting to ONNX...")
        onnx_path = output_dir / f"{args.model_name}.onnx"
        result = exporter.export_onnx(
            str(onnx_path),
            opset_version=14,
            simplify=True,
            verify=True,
        )
        results['onnx'] = result
        
        if result['success']:
            print(f"✓ ONNX export successful: {onnx_path}")
            print(f"  Size: {result['size_mb']:.2f} MB")
    
    if 'tflite' in args.formats:
        print("\n📦 Exporting to TFLite...")
        tflite_path = output_dir / f"{args.model_name}.tflite"
        result = exporter.export_tflite(
            str(tflite_path),
            quantize=False,
        )
        results['tflite'] = result
        
        if result.get('success'):
            print(f"✓ TFLite export successful: {tflite_path}")
            print(f"  Size: {result['size_mb']:.2f} MB")
        else:
            print(f"✗ TFLite export failed: {result.get('error')}")
    
    if 'tflite_quant' in args.formats:
        print("\n📦 Exporting to quantized TFLite...")
        tflite_quant_path = output_dir / f"{args.model_name}_quantized.tflite"
        result = exporter.export_tflite(
            str(tflite_quant_path),
            quantize=True,
            quantization_mode="dynamic",
        )
        results['tflite_quantized'] = result
        
        if result.get('success'):
            print(f"✓ Quantized TFLite export successful: {tflite_quant_path}")
            print(f"  Size: {result['size_mb']:.2f} MB")
        else:
            print(f"✗ Quantized TFLite export failed: {result.get('error')}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    
    for format_name, result in results.items():
        if result.get('success'):
            print(f"\n✓ {format_name.upper()}")
            print(f"  Path: {result['path']}")
            print(f"  Size: {result.get('size_mb', 0):.2f} MB")
        else:
            print(f"\n✗ {format_name.upper()}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
