"""Stable Diffusion inpainting wrapper with boundary-aware processing."""
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy

from watserface.types import VisionFrame
from watserface.inpainting.boundary import BoundaryDetector, create_boundary_detector
from watserface.inpainting.controlnet import ControlNetConditioner, create_controlnet_conditioner


class DiffusionInpainter:
    
    def __init__(
        self,
        model_id: str = 'runwayml/stable-diffusion-inpainting',
        device: str = 'auto',
        use_controlnet: bool = True
    ):
        self.model_id = model_id
        self.device = device
        self.use_controlnet = use_controlnet
        
        self.pipeline = None
        self.controlnet = None
        self.boundary_detector = create_boundary_detector()
        self.loaded = False
    
    def load(self) -> bool:
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline
            
            if self.device == 'auto':
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                elif torch.cuda.is_available():
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
            
            dtype = torch.float16 if self.device != 'cpu' else torch.float32
            
            self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None
            )
            self.pipeline.to(self.device)
            
            if self.use_controlnet:
                self.controlnet = create_controlnet_conditioner(self.device)
                self.controlnet.load_models()
            
            self.loaded = True
            return True
            
        except ImportError:
            print('diffusers not installed')
            return False
        except Exception as e:
            print(f'Failed to load diffusion model: {e}')
            return False
    
    def inpaint(
        self,
        frame: VisionFrame,
        mask: numpy.ndarray,
        prompt: str = 'realistic skin texture, natural lighting',
        negative_prompt: str = 'blurry, artifacts, distorted',
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        strength: float = 0.8,
        conditioning: Optional[Dict[str, Any]] = None
    ) -> VisionFrame:
        if not self.loaded:
            if not self.load():
                return self._fallback_inpaint(frame, mask)
        
        try:
            import torch
            from PIL import Image
            
            h, w = frame.shape[:2]
            
            target_size = (512, 512)
            frame_resized = cv2.resize(frame, target_size)
            mask_resized = cv2.resize(mask, target_size)
            
            image = Image.fromarray(frame_resized)
            
            if mask_resized.max() > 1:
                mask_resized = mask_resized.astype(numpy.float32) / 255.0
            mask_pil = Image.fromarray((mask_resized * 255).astype(numpy.uint8))
            
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    mask_image=mask_pil,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    strength=strength
                ).images[0]
            
            result_np = numpy.array(result)
            result_full = cv2.resize(result_np, (w, h))
            
            return result_full
            
        except Exception as e:
            print(f'Diffusion inpainting failed: {e}')
            return self._fallback_inpaint(frame, mask)
    
    def inpaint_with_boundary(
        self,
        frame: VisionFrame,
        xseg_mask: numpy.ndarray,
        face_mask: numpy.ndarray,
        depth_map: Optional[numpy.ndarray] = None,
        normal_map: Optional[numpy.ndarray] = None,
        **kwargs
    ) -> VisionFrame:
        boundary_zone, _ = self.boundary_detector.create_boundary_zone(xseg_mask)
        
        inpaint_mask = self.boundary_detector.create_inpaint_mask(
            xseg_mask, face_mask,
            boundary_expansion=15
        )
        
        conditioning = None
        if self.use_controlnet and self.controlnet and (depth_map is not None or normal_map is not None):
            conditioning = {
                'depth': depth_map,
                'normal': normal_map
            }
        
        inpainted = self.inpaint(
            frame, inpaint_mask,
            conditioning=conditioning,
            **kwargs
        )
        
        blend_mask = self.boundary_detector.create_feathered_blend_mask(inpaint_mask)
        
        if len(blend_mask.shape) == 2:
            blend_mask = numpy.stack([blend_mask] * 3, axis=-1)
        
        result = frame * (1 - blend_mask) + inpainted * blend_mask
        
        return result.astype(numpy.uint8)
    
    def _fallback_inpaint(
        self,
        frame: VisionFrame,
        mask: numpy.ndarray
    ) -> VisionFrame:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        mask_uint8 = mask.astype(numpy.uint8)
        if mask_uint8.max() <= 1:
            mask_uint8 = (mask_uint8 * 255).astype(numpy.uint8)
        
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        inpainted_bgr = cv2.inpaint(
            frame_bgr, mask_uint8,
            inpaintRadius=5,
            flags=cv2.INPAINT_TELEA
        )
        
        return cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)


class VideoInpainter:
    
    def __init__(self, inpainter: Optional[DiffusionInpainter] = None):
        self.inpainter = inpainter or DiffusionInpainter()
        self.prev_frame: Optional[VisionFrame] = None
        self.prev_result: Optional[VisionFrame] = None
        self.prev_flow: Optional[numpy.ndarray] = None
    
    def inpaint_frame(
        self,
        frame: VisionFrame,
        mask: numpy.ndarray,
        use_temporal: bool = True,
        **kwargs
    ) -> VisionFrame:
        result = self.inpainter.inpaint(frame, mask, **kwargs)
        
        if use_temporal and self.prev_result is not None:
            result = self._apply_temporal_consistency(frame, result, mask)
        
        self.prev_frame = frame.copy()
        self.prev_result = result.copy()
        
        return result
    
    def _apply_temporal_consistency(
        self,
        current_frame: VisionFrame,
        current_result: VisionFrame,
        mask: numpy.ndarray
    ) -> VisionFrame:
        if self.prev_frame is None or self.prev_result is None:
            return current_result
        
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        h, w = current_frame.shape[:2]
        y_coords, x_coords = numpy.mgrid[0:h, 0:w].astype(numpy.float32)
        new_x = x_coords + flow[:, :, 0]
        new_y = y_coords + flow[:, :, 1]
        
        warped_prev = cv2.remap(
            self.prev_result, new_x, new_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        temporal_weight = 0.3
        
        if len(mask.shape) == 2:
            mask_3ch = numpy.stack([mask] * 3, axis=-1)
        else:
            mask_3ch = mask
        
        if mask_3ch.max() > 1:
            mask_3ch = mask_3ch.astype(numpy.float32) / 255.0
        
        blended = current_result * (1 - temporal_weight * mask_3ch) + \
                  warped_prev * temporal_weight * mask_3ch
        
        self.prev_flow = flow
        
        return blended.astype(numpy.uint8)
    
    def reset(self) -> None:
        self.prev_frame = None
        self.prev_result = None
        self.prev_flow = None


def create_diffusion_inpainter(
    model_id: str = 'runwayml/stable-diffusion-inpainting',
    use_controlnet: bool = True
) -> DiffusionInpainter:
    return DiffusionInpainter(model_id=model_id, use_controlnet=use_controlnet)


def create_video_inpainter() -> VideoInpainter:
    return VideoInpainter()
