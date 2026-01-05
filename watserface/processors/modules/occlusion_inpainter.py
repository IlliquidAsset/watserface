from typing import Any, List, Optional
import cv2
import numpy

from watserface.types import ApplyStateItem, Args, Face, FaceSet, VisionFrame, AudioFrame

NAME = 'WATSERFACE.PROCESSORS.OCCLUSION_INPAINTER'

_inpainter = None
_video_inpainter = None


def get_inference_pool() -> Any:
    return None


def clear_inference_pool() -> None:
    global _inpainter, _video_inpainter
    _inpainter = None
    _video_inpainter = None


def register_args(program: argparse.ArgumentParser) -> None:
    pass


def apply_args(args: Args, apply_state_item: ApplyStateItem) -> None:
    pass


def pre_check() -> bool:
    return True


def pre_process(mode: str) -> bool:
    return True


def post_process() -> None:
    global _video_inpainter
    if _video_inpainter is not None:
        _video_inpainter.reset()


def get_reference_frame(source_face: Face, target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
    return temp_vision_frame


def _get_inpainter():
    global _inpainter
    if _inpainter is None:
        try:
            from watserface.inpainting.diffusion import create_diffusion_inpainter
            _inpainter = create_diffusion_inpainter(use_controlnet=True)
        except ImportError:
            _inpainter = None
    return _inpainter


def _get_video_inpainter():
    global _video_inpainter
    if _video_inpainter is None:
        try:
            from watserface.inpainting.diffusion import create_video_inpainter
            _video_inpainter = create_video_inpainter()
        except ImportError:
            _video_inpainter = None
    return _video_inpainter


def _extract_face_mask(face: Optional[Face], frame_shape: tuple) -> Optional[numpy.ndarray]:
    if face is None:
        return None
    
    h, w = frame_shape[:2]
    mask = numpy.zeros((h, w), dtype=numpy.float32)
    
    try:
        if hasattr(face, 'landmarks') and face.landmarks is not None:
            landmarks_68 = face.landmarks.get('68')
            if landmarks_68 is not None:
                points = landmarks_68.astype(numpy.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, 1.0)
                
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
                return mask
    except Exception:
        pass
        
    return None


def _get_depth_and_normal(frame: VisionFrame, face: Optional[Face]):
    depth_map = None
    normal_map = None
    
    try:
        from watserface.depth.estimator import estimate_depth
        depth_map = estimate_depth(frame)
    except ImportError:
        pass
    
    try:
        from watserface.face_helper import create_normal_map
        if face is not None and hasattr(face, 'landmarks'):
            landmarks_478 = face.landmarks.get('478')
            if landmarks_478 is not None:
                h, w = frame.shape[:2]
                normal_map = create_normal_map(landmarks_478, (w, h))
    except Exception:
        pass
    
    return depth_map, normal_map


def process_frame(inputs: Any) -> VisionFrame:
    target_vision_frame = inputs.get('target_vision_frame')
    
    if target_vision_frame is None:
        return inputs.get('target_vision_frame', numpy.zeros((1, 1, 3), dtype=numpy.uint8))
    
    xseg_mask = inputs.get('occlusion_mask')
    target_face = inputs.get('target_face')
    
    if xseg_mask is None or not numpy.any(xseg_mask > 0):
        return target_vision_frame
    
    inpainter = _get_inpainter()
    
    if inpainter is None:
        return _fallback_inpaint(target_vision_frame, xseg_mask)
    
    face_mask = _extract_face_mask(target_face, target_vision_frame.shape)
    depth_map, normal_map = _get_depth_and_normal(target_vision_frame, target_face)
    
    try:
        if face_mask is not None:
            result = inpainter.inpaint_with_boundary(
                target_vision_frame,
                xseg_mask,
                face_mask,
                depth_map=depth_map,
                normal_map=normal_map,
                prompt='realistic skin texture, seamless blend, natural lighting',
                negative_prompt='blurry, artifacts, distorted, unnatural',
                num_inference_steps=20,
                strength=0.75
            )
        else:
            result = inpainter.inpaint(
                target_vision_frame,
                xseg_mask,
                prompt='realistic skin texture, natural lighting',
                num_inference_steps=20,
                strength=0.75
            )
        
        return result
        
    except Exception as e:
        print(f'Diffusion inpainting failed: {e}')
        return _fallback_inpaint(target_vision_frame, xseg_mask)


def process_frame_video(inputs: Any) -> VisionFrame:
    target_vision_frame = inputs.get('target_vision_frame')
    xseg_mask = inputs.get('occlusion_mask')
    
    if target_vision_frame is None:
        return numpy.zeros((1, 1, 3), dtype=numpy.uint8)
    
    if xseg_mask is None or not numpy.any(xseg_mask > 0):
        return target_vision_frame
    
    video_inpainter = _get_video_inpainter()
    
    if video_inpainter is None:
        return process_frame(inputs)
    
    try:
        result = video_inpainter.inpaint_frame(
            target_vision_frame,
            xseg_mask,
            use_temporal=True,
            prompt='realistic skin texture, seamless blend',
            num_inference_steps=15,
            strength=0.7
        )
        return result
    except Exception:
        return process_frame(inputs)


def _fallback_inpaint(frame: VisionFrame, mask: numpy.ndarray) -> VisionFrame:
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


def process_frames(source_paths: List[str], temp_frame_paths: List[str], update: Any) -> None:
    pass


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    pass


def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
    pass
