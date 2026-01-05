"""ONNX model conversion and optimized inference for depth and inpainting models."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import numpy

from watserface.types import VisionFrame


@dataclass
class ConversionConfig:
    """Configuration for ONNX model conversion."""
    opset_version: int = 14
    optimize: bool = True
    quantize: bool = False
    quantization_mode: str = 'dynamic'  # 'dynamic', 'static', 'qat'
    input_shapes: Optional[Dict[str, List[int]]] = None
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None


class ONNXConverter:
    """Converts PyTorch models to ONNX format with optimization."""
    
    def __init__(self, output_dir: str = '.models/onnx'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_pytorch_model(
        self,
        model: Any,
        model_name: str,
        input_shape: Tuple[int, ...],
        config: Optional[ConversionConfig] = None
    ) -> Optional[str]:
        """Convert a PyTorch model to ONNX format."""
        config = config or ConversionConfig()
        
        try:
            import torch
            import torch.onnx
            
            output_path = self.output_dir / f'{model_name}.onnx'
            
            model.eval()
            
            dummy_input = torch.randn(*input_shape)
            if hasattr(model, 'device'):
                dummy_input = dummy_input.to(model.device)
            
            dynamic_axes = config.dynamic_axes or {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
            
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                opset_version=config.opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                do_constant_folding=True
            )
            
            if config.optimize:
                self._optimize_onnx(output_path)
            
            if config.quantize:
                self._quantize_onnx(output_path, config.quantization_mode)
            
            return str(output_path)
            
        except ImportError:
            print('PyTorch or ONNX not installed')
            return None
        except Exception as e:
            print(f'Failed to convert model: {e}')
            return None
    
    def convert_midas_model(
        self,
        model_type: str = 'midas_small',
        input_size: Tuple[int, int] = (384, 384)
    ) -> Optional[str]:
        """Convert MiDaS depth estimation model to ONNX."""
        try:
            import torch
            
            if model_type == 'midas_small':
                model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
            elif model_type == 'midas_large':
                model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large', trust_repo=True)
            else:
                model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
            
            model.eval()
            
            input_shape = (1, 3, input_size[0], input_size[1])
            
            config = ConversionConfig(
                opset_version=14,
                optimize=True,
                dynamic_axes={
                    'input': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 1: 'height', 2: 'width'}
                }
            )
            
            return self.convert_pytorch_model(
                model, f'midas_{model_type}', input_shape, config
            )
            
        except Exception as e:
            print(f'Failed to convert MiDaS model: {e}')
            return None
    
    def convert_vae_encoder(
        self,
        model_id: str = 'runwayml/stable-diffusion-inpainting'
    ) -> Optional[str]:
        """Convert SD VAE encoder to ONNX for fast latent encoding."""
        try:
            import torch
            from diffusers import AutoencoderKL
            
            vae = AutoencoderKL.from_pretrained(
                model_id,
                subfolder='vae',
                torch_dtype=torch.float32
            )
            vae.eval()
            
            class VAEEncoder(torch.nn.Module):
                def __init__(self, vae):
                    super().__init__()
                    self.encoder = vae.encoder
                    self.quant_conv = vae.quant_conv
                
                def forward(self, x):
                    h = self.encoder(x)
                    moments = self.quant_conv(h)
                    mean, _ = torch.chunk(moments, 2, dim=1)
                    return mean * 0.18215
            
            encoder = VAEEncoder(vae)
            encoder.eval()
            
            input_shape = (1, 3, 512, 512)
            config = ConversionConfig(
                opset_version=14,
                optimize=True,
                dynamic_axes={
                    'input': {0: 'batch', 2: 'height', 3: 'width'},
                    'output': {0: 'batch', 2: 'height', 3: 'width'}
                }
            )
            
            return self.convert_pytorch_model(
                encoder, 'sd_vae_encoder', input_shape, config
            )
            
        except Exception as e:
            print(f'Failed to convert VAE encoder: {e}')
            return None
    
    def _optimize_onnx(self, model_path: Path) -> None:
        """Apply ONNX graph optimizations."""
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            optimized_path = model_path.with_suffix('.opt.onnx')
            
            optimized_model = optimizer.optimize_model(
                str(model_path),
                model_type='bert',  # Generic optimization
                num_heads=0,
                hidden_size=0,
                optimization_options=None
            )
            optimized_model.save_model_to_file(str(optimized_path))
            
            os.replace(optimized_path, model_path)
            
        except ImportError:
            try:
                import onnx
                from onnx import optimizer as onnx_optimizer
                
                model = onnx.load(str(model_path))
                
                passes = [
                    'eliminate_deadend',
                    'eliminate_identity',
                    'eliminate_nop_dropout',
                    'eliminate_nop_monotone_argmax',
                    'eliminate_nop_pad',
                    'eliminate_nop_transpose',
                    'eliminate_unused_initializer',
                    'extract_constant_to_initializer',
                    'fuse_add_bias_into_conv',
                    'fuse_bn_into_conv',
                    'fuse_consecutive_concats',
                    'fuse_consecutive_reduce_unsqueeze',
                    'fuse_consecutive_squeezes',
                    'fuse_consecutive_transposes',
                    'fuse_matmul_add_bias_into_gemm',
                    'fuse_pad_into_conv',
                    'fuse_transpose_into_gemm'
                ]
                
                optimized_model = onnx_optimizer.optimize(model, passes)
                onnx.save(optimized_model, str(model_path))
                
            except Exception:
                pass
    
    def _quantize_onnx(self, model_path: Path, mode: str = 'dynamic') -> None:
        """Apply quantization to ONNX model."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            quantized_path = model_path.with_suffix('.quant.onnx')
            
            quantize_dynamic(
                str(model_path),
                str(quantized_path),
                weight_type=QuantType.QUInt8
            )
            
            os.replace(quantized_path, model_path)
            
        except ImportError:
            print('onnxruntime quantization not available')
        except Exception as e:
            print(f'Quantization failed: {e}')


class ONNXInferenceSession:
    """Optimized ONNX inference session with provider selection."""
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None
    ):
        self.model_path = model_path
        self.session = None
        self.providers = providers
        self.input_name = None
        self.output_name = None
    
    def load(self) -> bool:
        """Load ONNX model and create inference session."""
        try:
            import onnxruntime as ort
            
            if self.providers is None:
                available = ort.get_available_providers()
                self.providers = []
                
                if 'CoreMLExecutionProvider' in available:
                    self.providers.append('CoreMLExecutionProvider')
                if 'CUDAExecutionProvider' in available:
                    self.providers.append('CUDAExecutionProvider')
                
                self.providers.append('CPUExecutionProvider')
            
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=self.providers
            )
            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            return True
            
        except ImportError:
            print('onnxruntime not installed')
            return False
        except Exception as e:
            print(f'Failed to load ONNX session: {e}')
            return False
    
    def run(self, input_data: numpy.ndarray) -> Optional[numpy.ndarray]:
        """Run inference on input data."""
        if self.session is None:
            if not self.load():
                return None
        
        try:
            input_data = input_data.astype(numpy.float32)
            
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            
            return outputs[0]
            
        except Exception as e:
            print(f'Inference failed: {e}')
            return None
    
    def run_batch(
        self,
        inputs: List[numpy.ndarray],
        batch_size: int = 4
    ) -> List[numpy.ndarray]:
        """Run inference on batched inputs."""
        results = []
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_array = numpy.stack(batch, axis=0)
            
            output = self.run(batch_array)
            if output is not None:
                for j in range(output.shape[0]):
                    results.append(output[j])
        
        return results
    
    def get_input_shape(self) -> Optional[List[int]]:
        """Get expected input shape."""
        if self.session is None:
            return None
        return self.session.get_inputs()[0].shape
    
    def get_output_shape(self) -> Optional[List[int]]:
        """Get expected output shape."""
        if self.session is None:
            return None
        return self.session.get_outputs()[0].shape
    
    def benchmark(self, input_shape: Tuple[int, ...], num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference performance."""
        import time
        
        dummy_input = numpy.random.randn(*input_shape).astype(numpy.float32)
        
        for _ in range(10):
            self.run(dummy_input)
        
        times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            self.run(dummy_input)
            times.append(time.perf_counter() - start)
        
        return {
            'mean_ms': numpy.mean(times) * 1000,
            'std_ms': numpy.std(times) * 1000,
            'min_ms': numpy.min(times) * 1000,
            'max_ms': numpy.max(times) * 1000,
            'fps': 1.0 / numpy.mean(times)
        }


class OptimizedDepthEstimator:
    """ONNX-accelerated depth estimation."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.session: Optional[ONNXInferenceSession] = None
        self.input_size = (384, 384)
    
    def load(self, model_path: Optional[str] = None) -> bool:
        """Load ONNX depth model."""
        path = model_path or self.model_path
        
        if path is None:
            converter = ONNXConverter()
            path = converter.convert_midas_model('midas_small', self.input_size)
        
        if path is None:
            return False
        
        self.model_path = path
        self.session = ONNXInferenceSession(path)
        return self.session.load()
    
    def estimate(self, frame: VisionFrame) -> VisionFrame:
        """Estimate depth from frame using ONNX model."""
        if self.session is None:
            if not self.load():
                return numpy.zeros(frame.shape[:2], dtype=numpy.float32)
        
        import cv2
        
        h, w = frame.shape[:2]
        
        resized = cv2.resize(frame, self.input_size)
        
        if resized.shape[-1] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        input_data = resized.astype(numpy.float32) / 255.0
        input_data = numpy.transpose(input_data, (2, 0, 1))
        input_data = numpy.expand_dims(input_data, 0)
        
        output = self.session.run(input_data)
        
        if output is None:
            return numpy.zeros((h, w), dtype=numpy.float32)
        
        depth = output[0]
        if len(depth.shape) == 3:
            depth = depth[0]
        
        depth = cv2.resize(depth, (w, h))
        
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)
        
        return depth.astype(numpy.float32)


def convert_depth_model(
    model_type: str = 'midas_small',
    output_dir: str = '.models/onnx'
) -> Optional[str]:
    """Convenience function to convert depth model to ONNX."""
    converter = ONNXConverter(output_dir)
    return converter.convert_midas_model(model_type)


def convert_inpainting_encoder(
    model_id: str = 'runwayml/stable-diffusion-inpainting',
    output_dir: str = '.models/onnx'
) -> Optional[str]:
    """Convenience function to convert VAE encoder to ONNX."""
    converter = ONNXConverter(output_dir)
    return converter.convert_vae_encoder(model_id)


def create_onnx_converter(output_dir: str = '.models/onnx') -> ONNXConverter:
    """Create ONNX converter instance."""
    return ONNXConverter(output_dir)


def create_inference_session(model_path: str) -> ONNXInferenceSession:
    """Create ONNX inference session."""
    return ONNXInferenceSession(model_path)
