"""
Export Module - Model Export to ONNX and TFLite

Exports trained PyTorch models to ONNX and TFLite formats for multi-platform deployment.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import numpy as np


class ModelExporter:
    """
    Export PyTorch models to ONNX and TFLite formats.
    
    Features:
    - ONNX export with optimization
    - TFLite export with quantization options
    - Model validation after export
    - Metadata embedding
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = "cpu",
    ):
        """
        Initialize exporter.
        
        Args:
            model: PyTorch model to export
            input_shape: Input shape (batch_size, features)
            device: Device to run model on
        """
        self.model = model.to(device).eval()
        self.input_shape = input_shape
        self.device = device
    
    def export_onnx(
        self,
        output_path: str,
        opset_version: int = 14,
        simplify: bool = True,
        verify: bool = True,
    ) -> Dict[str, Any]:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            simplify: Whether to simplify the ONNX graph
            verify: Whether to verify the exported model
            
        Returns:
            Dictionary with export status and metadata
        """
        import onnx
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            },
        )
        
        # Simplify if requested
        if simplify:
            try:
                from onnxsim import simplify as onnx_simplify
                
                # Load model
                onnx_model = onnx.load(str(output_path))
                
                # Simplify
                model_simp, check = onnx_simplify(onnx_model)
                
                if check:
                    # Save simplified model
                    onnx.save(model_simp, str(output_path))
                    print("ONNX model simplified successfully")
                else:
                    print("Warning: ONNX simplification check failed")
            except ImportError:
                print("onnx-simplifier not installed. Skipping simplification.")
        
        # Verify if requested
        if verify:
            self._verify_onnx(output_path, dummy_input)
        
        # Get model info
        onnx_model = onnx.load(str(output_path))
        model_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        return {
            'success': True,
            'format': 'ONNX',
            'path': str(output_path),
            'opset_version': opset_version,
            'size_mb': model_size,
        }
    
    def _verify_onnx(
        self,
        onnx_path: Path,
        test_input: torch.Tensor,
    ):
        """
        Verify ONNX export by comparing outputs.
        
        Args:
            onnx_path: Path to ONNX model
            test_input: Test input tensor
        """
        import onnx
        import onnxruntime as ort
        
        # Check ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        
        # Run ONNX inference
        ort_session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Run PyTorch inference
        with torch.no_grad():
            torch_output = self.model(test_input).cpu().numpy()
        
        # Compare outputs
        max_diff = np.max(np.abs(ort_outputs[0] - torch_output))
        
        if max_diff < 1e-5:
            print(f"✓ ONNX export verified (max diff: {max_diff:.2e})")
        else:
            print(f"⚠ Warning: ONNX output differs (max diff: {max_diff:.2e})")
    
    def export_tflite(
        self,
        output_path: str,
        quantize: bool = False,
        quantization_mode: str = "dynamic",
    ) -> Dict[str, Any]:
        """
        Export model to TFLite format.
        
        Args:
            output_path: Path to save TFLite model
            quantize: Whether to quantize the model
            quantization_mode: Quantization mode ('dynamic', 'int8', 'float16')
            
        Returns:
            Dictionary with export status and metadata
        """
        try:
            import tensorflow as tf
            import tf2onnx
        except ImportError:
            return {
                'success': False,
                'error': 'TensorFlow not installed. Install with: pip install tensorflow tf2onnx',
            }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # First export to ONNX
        onnx_path = output_path.with_suffix('.onnx')
        self.export_onnx(str(onnx_path), verify=False, simplify=False)
        
        # Convert ONNX to TFLite via TensorFlow SavedModel
        # This is a two-step process: ONNX -> TF SavedModel -> TFLite
        
        # Step 1: Convert ONNX to TF SavedModel
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        
        saved_model_path = output_path.parent / "saved_model"
        tf_rep.export_graph(str(saved_model_path))
        
        # Step 2: Convert SavedModel to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
        
        # Apply quantization if requested
        if quantize:
            if quantization_mode == "dynamic":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            elif quantization_mode == "int8":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            elif quantization_mode == "float16":
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Clean up temporary files
        import shutil
        onnx_path.unlink()
        if saved_model_path.exists():
            shutil.rmtree(saved_model_path)
        
        model_size = output_path.stat().st_size / (1024 * 1024)  # MB
        
        return {
            'success': True,
            'format': 'TFLite',
            'path': str(output_path),
            'quantized': quantize,
            'quantization_mode': quantization_mode if quantize else None,
            'size_mb': model_size,
        }
    
    def export_all(
        self,
        output_dir: str,
        model_name: str = "gesture_model",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Export model to all supported formats.
        
        Args:
            output_dir: Directory to save exported models
            model_name: Base name for exported models
            
        Returns:
            Dictionary mapping format names to export results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Export ONNX
        print("\n📦 Exporting to ONNX...")
        onnx_path = output_path / f"{model_name}.onnx"
        results['onnx'] = self.export_onnx(str(onnx_path))
        
        # Export TFLite
        print("\n📦 Exporting to TFLite...")
        tflite_path = output_path / f"{model_name}.tflite"
        results['tflite'] = self.export_tflite(str(tflite_path))
        
        # Export quantized TFLite
        print("\n📦 Exporting to quantized TFLite...")
        tflite_quant_path = output_path / f"{model_name}_quantized.tflite"
        results['tflite_quantized'] = self.export_tflite(
            str(tflite_quant_path),
            quantize=True,
            quantization_mode="dynamic",
        )
        
        # Print summary
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
                print(f"  Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 60)
        
        return results


def export_with_metadata(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    metadata: Dict[str, Any],
):
    """
    Export model with embedded metadata.
    
    Args:
        model: PyTorch model
        output_path: Path to save model
        input_shape: Input shape
        metadata: Metadata dictionary to embed
    """
    exporter = ModelExporter(model, input_shape)
    
    # Export ONNX
    result = exporter.export_onnx(output_path)
    
    if result['success']:
        import onnx
        
        # Load ONNX model
        onnx_model = onnx.load(output_path)
        
        # Add metadata
        for key, value in metadata.items():
            meta = onnx_model.metadata_props.add()
            meta.key = key
            meta.value = str(value)
        
        # Save with metadata
        onnx.save(onnx_model, output_path)
        
        print(f"✓ Model exported with metadata: {output_path}")
    
    return result
