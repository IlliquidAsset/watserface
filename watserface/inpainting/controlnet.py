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


def create_controlnet_conditioner(device: str = 'auto') -> ControlNetConditioner:
    return ControlNetConditioner(device=device)


def create_conditioning_pipeline() -> ConditioningPipeline:
    return ConditioningPipeline()
