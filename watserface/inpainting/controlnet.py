"""ControlNet integration for conditioning diffusion inpainting with normal maps and depth."""
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy

from watserface.types import VisionFrame
from watserface.face_helper import create_normal_map
from watserface.depth.estimator import estimate_depth


class ControlNetConditioner:
    
    def __init__(self, device: str = 'auto'):
        self.device = device
        self.controlnet_depth = None
        self.controlnet_normal = None
        self.loaded = False
    
    def load_models(self) -> bool:
        try:
            import torch
            from diffusers import ControlNetModel
            
            if self.device == 'auto':
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                elif torch.cuda.is_available():
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            
            self.controlnet_depth = ControlNetModel.from_pretrained(
                'lllyasviel/control_v11f1p_sd15_depth',
                torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32
            )
            
            self.controlnet_normal = ControlNetModel.from_pretrained(
                'lllyasviel/control_v11p_sd15_normalbae',
                torch_dtype=torch.float16 if self.device != 'cpu' else torch.float32
            )
            
            self.loaded = True
            return True
            
        except ImportError:
            print('diffusers not installed, ControlNet unavailable')
            return False
        except Exception as e:
            print(f'Failed to load ControlNet models: {e}')
            return False
    
    def prepare_depth_conditioning(
        self,
        frame: VisionFrame,
        depth_map: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        if depth_map is None:
            depth_map = estimate_depth(frame)
        
        depth_normalized = (depth_map * 255).astype(numpy.uint8)
        depth_3ch = numpy.stack([depth_normalized] * 3, axis=-1)
        
        return depth_3ch
    
    def prepare_normal_conditioning(
        self,
        landmarks_478: numpy.ndarray,
        frame_size: Tuple[int, int]
    ) -> numpy.ndarray:
        normal_map = create_normal_map(landmarks_478, frame_size)
        return normal_map
    
    def prepare_combined_conditioning(
        self,
        frame: VisionFrame,
        landmarks_478: Optional[numpy.ndarray] = None,
        depth_map: Optional[numpy.ndarray] = None,
        depth_weight: float = 0.5,
        normal_weight: float = 0.5
    ) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        
        depth_cond = self.prepare_depth_conditioning(frame, depth_map)
        
        if landmarks_478 is not None:
            normal_cond = self.prepare_normal_conditioning(landmarks_478, (w, h))
        else:
            normal_cond = numpy.zeros((h, w, 3), dtype=numpy.uint8)
            normal_cond[:, :, 2] = 128
        
        return {
            'depth': depth_cond,
            'normal': normal_cond,
            'depth_weight': depth_weight,
            'normal_weight': normal_weight,
            'controlnets': [self.controlnet_depth, self.controlnet_normal] if self.loaded else None,
            'conditioning_scales': [depth_weight, normal_weight]
        }
    
    def apply_conditioning_to_pipeline(
        self,
        pipeline: Any,
        conditioning: Dict[str, Any]
    ) -> Any:
        if not self.loaded or conditioning.get('controlnets') is None:
            return pipeline
        
        try:
            from diffusers import StableDiffusionControlNetInpaintPipeline
            
            pipeline.controlnet = conditioning['controlnets']
            
        except Exception as e:
            print(f'Failed to apply ControlNet conditioning: {e}')
        
        return pipeline
    
    def preprocess_for_controlnet(
        self,
        image: numpy.ndarray,
        target_size: Tuple[int, int] = (512, 512)
    ) -> numpy.ndarray:
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        if image.max() > 1:
            image = image.astype(numpy.float32) / 255.0
        
        return image


class ConditioningPipeline:
    
    def __init__(self):
        self.conditioner = ControlNetConditioner()
        self.depth_estimator_loaded = False
    
    def prepare_frame_conditioning(
        self,
        frame: VisionFrame,
        face_landmarks: Optional[numpy.ndarray] = None,
        face_mask: Optional[numpy.ndarray] = None,
        use_depth: bool = True,
        use_normal: bool = True
    ) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        result = {
            'frame': frame,
            'frame_size': (w, h)
        }
        
        if use_depth:
            depth = estimate_depth(frame)
            result['depth_map'] = depth
            result['depth_conditioning'] = self.conditioner.prepare_depth_conditioning(frame, depth)
        
        if use_normal and face_landmarks is not None:
            result['normal_conditioning'] = self.conditioner.prepare_normal_conditioning(
                face_landmarks, (w, h)
            )
        
        if face_mask is not None:
            result['face_mask'] = face_mask
        
        return result
    
    def blend_with_conditioning(
        self,
        original: VisionFrame,
        inpainted: VisionFrame,
        mask: numpy.ndarray,
        depth_map: Optional[numpy.ndarray] = None,
        blend_mode: str = 'depth_aware'
    ) -> VisionFrame:
        if mask.max() > 1:
            mask = mask.astype(numpy.float32) / 255.0
        
        if len(mask.shape) == 2:
            mask = numpy.stack([mask] * 3, axis=-1)
        
        if blend_mode == 'simple':
            blended = original * (1 - mask) + inpainted * mask
            
        elif blend_mode == 'depth_aware' and depth_map is not None:
            depth_normalized = depth_map.copy()
            if depth_normalized.max() > 1:
                depth_normalized = depth_normalized / depth_normalized.max()
            
            if len(depth_normalized.shape) == 2:
                depth_normalized = numpy.stack([depth_normalized] * 3, axis=-1)
            
            blend_weight = mask * (0.5 + 0.5 * depth_normalized)
            blend_weight = numpy.clip(blend_weight, 0, 1)
            
            blended = original * (1 - blend_weight) + inpainted * blend_weight
            
        elif blend_mode == 'feathered':
            kernel_size = max(3, int(min(mask.shape[:2]) * 0.05) | 1)
            feathered_mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
            blended = original * (1 - feathered_mask) + inpainted * feathered_mask
            
        else:
            blended = original * (1 - mask) + inpainted * mask
        
        return blended.astype(numpy.uint8)


class ControlNetPipeline:
    
    DEFAULT_DEPTH_MODEL = 'diffusers/controlnet-depth-sdxl-1.0-small'
    DEFAULT_CANNY_MODEL = 'diffusers/controlnet-canny-sdxl-1.0'
    DEFAULT_BASE_MODEL = 'stabilityai/stable-diffusion-xl-base-1.0'
    
    def __init__(
        self,
        device: str = 'auto',
        controlnet_conditioning_scale: float = 0.75,
        depth_model_id: Optional[str] = None,
        canny_model_id: Optional[str] = None
    ):
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.depth_model_id = depth_model_id or self.DEFAULT_DEPTH_MODEL
        self.canny_model_id = canny_model_id or self.DEFAULT_CANNY_MODEL
        self.loaded = False
        self.pipeline = None
        self.controlnet_depth = None
        self.controlnet_canny = None
        
        self._resolve_device(device)
    
    def _resolve_device(self, device: str) -> None:
        if device == 'auto':
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                elif torch.cuda.is_available():
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device
    
    def load(self) -> bool:
        try:
            import torch
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
            
            dtype = torch.float16 if self.device != 'cpu' else torch.float32
            
            self.controlnet_depth = ControlNetModel.from_pretrained(
                self.depth_model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
            
            self.controlnet_canny = ControlNetModel.from_pretrained(
                self.canny_model_id,
                torch_dtype=dtype,
                use_safetensors=True
            )
            
            vae = AutoencoderKL.from_pretrained(
                'madebyollin/sdxl-vae-fp16-fix',
                torch_dtype=dtype
            )
            
            self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.DEFAULT_BASE_MODEL,
                controlnet=[self.controlnet_depth, self.controlnet_canny],
                vae=vae,
                torch_dtype=dtype,
                use_safetensors=True
            )
            self.pipeline.to(self.device)
            
            if self.device != 'cpu':
                self.pipeline.enable_model_cpu_offload()
            
            self.loaded = True
            return True
            
        except ImportError as e:
            print(f'diffusers not installed: {e}')
            return False
        except Exception as e:
            print(f'Failed to load ControlNet pipeline: {e}')
            return False
    
    def prepare_depth_conditioning(
        self,
        image: numpy.ndarray,
        target_size: Tuple[int, int] = (512, 512)
    ) -> numpy.ndarray:
        if image.shape[:2] != target_size[::-1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        depth_map = estimate_depth(image)
        
        if depth_map.max() > 0:
            depth_normalized = (depth_map / depth_map.max() * 255).astype(numpy.uint8)
        else:
            depth_normalized = numpy.zeros(depth_map.shape, dtype=numpy.uint8)
        
        if len(depth_normalized.shape) == 2:
            depth_3ch = numpy.stack([depth_normalized] * 3, axis=-1)
        else:
            depth_3ch = depth_normalized
        
        return depth_3ch
    
    def prepare_canny_conditioning(
        self,
        image: numpy.ndarray,
        target_size: Tuple[int, int] = (512, 512),
        low_threshold: int = 100,
        high_threshold: int = 200
    ) -> numpy.ndarray:
        if image.shape[:2] != target_size[::-1]:
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        edges_3ch = numpy.stack([edges] * 3, axis=-1)
        
        return edges_3ch
    
    def process(
        self,
        image: numpy.ndarray,
        prompt: str = 'high quality face, realistic skin texture',
        negative_prompt: str = 'blurry, distorted, low quality',
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5
    ) -> numpy.ndarray:
        if not self.loaded:
            return self._fallback_process(image)
        
        return self._run_pipeline(
            image, prompt, negative_prompt,
            num_inference_steps, guidance_scale
        )
    
    def _run_pipeline(
        self,
        image: numpy.ndarray,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float
    ) -> numpy.ndarray:
        try:
            import torch
            from PIL import Image
            
            target_size = (512, 512)
            
            depth_cond = self.prepare_depth_conditioning(image, target_size)
            canny_cond = self.prepare_canny_conditioning(image, target_size)
            
            depth_pil = Image.fromarray(depth_cond)
            canny_pil = Image.fromarray(canny_cond)
            
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=[depth_pil, canny_pil],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=[
                        self.controlnet_conditioning_scale,
                        self.controlnet_conditioning_scale
                    ]
                ).images[0]
            
            return numpy.array(result)
            
        except Exception as e:
            print(f'Pipeline processing failed: {e}')
            return self._fallback_process(image)
    
    def _fallback_process(self, image: numpy.ndarray) -> numpy.ndarray:
        target_size = (512, 512)
        if image.shape[:2] != target_size[::-1]:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        return image


def create_controlnet_pipeline(
    device: str = 'auto',
    controlnet_conditioning_scale: float = 0.75
) -> ControlNetPipeline:
    return ControlNetPipeline(
        device=device,
        controlnet_conditioning_scale=controlnet_conditioning_scale
    )


def create_controlnet_conditioner(device: str = 'auto') -> ControlNetConditioner:
    return ControlNetConditioner(device=device)


def create_conditioning_pipeline() -> ConditioningPipeline:
    return ConditioningPipeline()
