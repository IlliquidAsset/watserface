"""Mask annotation component using Gradio ImageEditor for manual XSeg mask editing."""
import gradio
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy

from watserface.studio.occlusion_trainer import OcclusionTrainer, create_occlusion_trainer
from watserface.types import VisionFrame


ANNOTATOR_IMAGE: Optional[gradio.ImageEditor] = None
ANNOTATOR_PREVIEW: Optional[gradio.Image] = None
ANNOTATOR_MODE: Optional[gradio.Radio] = None
ANNOTATOR_FRAME_SLIDER: Optional[gradio.Slider] = None
ANNOTATOR_GENERATE_BTN: Optional[gradio.Button] = None
ANNOTATOR_APPLY_BTN: Optional[gradio.Button] = None
ANNOTATOR_STATUS: Optional[gradio.Textbox] = None

CURRENT_FRAME: Optional[VisionFrame] = None
CURRENT_MASK: Optional[numpy.ndarray] = None
OCCLUSION_TRAINER: Optional[OcclusionTrainer] = None


def pre_check() -> bool:
    return True


def render() -> None:
    global ANNOTATOR_IMAGE, ANNOTATOR_PREVIEW, ANNOTATOR_MODE
    global ANNOTATOR_FRAME_SLIDER, ANNOTATOR_GENERATE_BTN, ANNOTATOR_APPLY_BTN
    global ANNOTATOR_STATUS, OCCLUSION_TRAINER
    
    OCCLUSION_TRAINER = create_occlusion_trainer()
    
    with gradio.Column():
        gradio.Markdown('### Mask Annotator')
        
        with gradio.Row():
            ANNOTATOR_MODE = gradio.Radio(
                label='Edit Mode',
                choices=['add', 'subtract', 'replace'],
                value='add'
            )
        
        with gradio.Row():
            ANNOTATOR_GENERATE_BTN = gradio.Button(
                'Auto-Generate Mask',
                variant='secondary'
            )
            ANNOTATOR_APPLY_BTN = gradio.Button(
                'Apply Annotations',
                variant='primary'
            )
        
        with gradio.Row():
            with gradio.Column(scale=1):
                ANNOTATOR_IMAGE = gradio.ImageEditor(
                    label='Draw on Image',
                    type='numpy',
                    brush=gradio.Brush(
                        colors=['#FFFFFF'],
                        default_size=20,
                        color_mode='fixed'
                    ),
                    eraser=gradio.Eraser(default_size=20),
                    height=512,
                    width=512
                )
            
            with gradio.Column(scale=1):
                ANNOTATOR_PREVIEW = gradio.Image(
                    label='Mask Preview',
                    type='numpy',
                    interactive=False,
                    height=512,
                    width=512
                )
        
        ANNOTATOR_STATUS = gradio.Textbox(
            label='Status',
            value='Ready. Load an image to begin annotation.',
            interactive=False
        )


def listen() -> None:
    if ANNOTATOR_GENERATE_BTN:
        ANNOTATOR_GENERATE_BTN.click(
            fn=handle_auto_generate,
            inputs=[ANNOTATOR_IMAGE],
            outputs=[ANNOTATOR_IMAGE, ANNOTATOR_PREVIEW, ANNOTATOR_STATUS]
        )
    
    if ANNOTATOR_APPLY_BTN:
        ANNOTATOR_APPLY_BTN.click(
            fn=handle_apply_annotations,
            inputs=[ANNOTATOR_IMAGE, ANNOTATOR_MODE],
            outputs=[ANNOTATOR_PREVIEW, ANNOTATOR_STATUS]
        )
    
    if ANNOTATOR_IMAGE:
        ANNOTATOR_IMAGE.change(
            fn=handle_image_change,
            inputs=[ANNOTATOR_IMAGE],
            outputs=[ANNOTATOR_PREVIEW, ANNOTATOR_STATUS]
        )


def handle_auto_generate(image_data: Optional[Dict[str, Any]]) -> Tuple[Any, Any, str]:
    global CURRENT_FRAME, CURRENT_MASK
    
    if image_data is None:
        return None, None, 'No image loaded'
    
    background = image_data.get('background')
    if background is None:
        return image_data, None, 'No background image found'
    
    if isinstance(background, numpy.ndarray):
        frame = background
    else:
        return image_data, None, 'Invalid image format'
    
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
    
    CURRENT_FRAME = frame
    
    mask_result = OCCLUSION_TRAINER.generate_mask_for_image(frame)
    
    if mask_result is None:
        return image_data, None, 'No face detected in image'
    
    CURRENT_MASK = mask_result.mask
    
    preview = OCCLUSION_TRAINER.get_mask_preview(frame, CURRENT_MASK)
    
    mask_layer = numpy.zeros_like(frame)
    mask_3ch = numpy.stack([CURRENT_MASK] * 3, axis=-1) / 255.0
    overlay_frame = frame.copy()
    
    return {
        'background': frame,
        'layers': [mask_layer],
        'composite': overlay_frame
    }, preview, f'Auto-generated mask (confidence: {mask_result.confidence:.2f})'


def handle_apply_annotations(
    image_data: Optional[Dict[str, Any]],
    mode: str
) -> Tuple[Any, str]:
    global CURRENT_MASK
    
    if image_data is None:
        return None, 'No image data'
    
    if CURRENT_FRAME is None:
        return None, 'No frame loaded. Generate a mask first.'
    
    layers = image_data.get('layers', [])
    composite = image_data.get('composite')
    
    annotation_mask = None
    
    if layers:
        for layer in layers:
            if isinstance(layer, numpy.ndarray) and layer.shape[:2] == CURRENT_FRAME.shape[:2]:
                if len(layer.shape) == 3:
                    gray = cv2.cvtColor(layer, cv2.COLOR_RGB2GRAY)
                else:
                    gray = layer
                
                if annotation_mask is None:
                    annotation_mask = gray
                else:
                    annotation_mask = cv2.bitwise_or(annotation_mask, gray)
    
    if annotation_mask is None and composite is not None:
        if isinstance(composite, numpy.ndarray):
            diff = cv2.absdiff(composite, CURRENT_FRAME)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
            _, annotation_mask = cv2.threshold(gray_diff, 10, 255, cv2.THRESH_BINARY)
    
    if annotation_mask is None:
        return OCCLUSION_TRAINER.get_mask_preview(CURRENT_FRAME, CURRENT_MASK) if CURRENT_MASK is not None else None, \
            'No annotations detected'
    
    if CURRENT_MASK is None:
        CURRENT_MASK = numpy.zeros(CURRENT_FRAME.shape[:2], dtype=numpy.uint8)
    
    CURRENT_MASK = OCCLUSION_TRAINER.refine_mask_with_annotation(
        CURRENT_MASK,
        annotation_mask,
        mode
    )
    
    preview = OCCLUSION_TRAINER.get_mask_preview(CURRENT_FRAME, CURRENT_MASK)
    
    return preview, f'Annotations applied in {mode} mode'


def handle_image_change(image_data: Optional[Dict[str, Any]]) -> Tuple[Any, str]:
    global CURRENT_FRAME
    
    if image_data is None:
        return None, 'No image'
    
    background = image_data.get('background')
    if background is None:
        return None, 'No background image'
    
    if isinstance(background, numpy.ndarray):
        CURRENT_FRAME = background
        
        if CURRENT_MASK is not None and CURRENT_MASK.shape[:2] == CURRENT_FRAME.shape[:2]:
            preview = OCCLUSION_TRAINER.get_mask_preview(CURRENT_FRAME, CURRENT_MASK)
            return preview, 'Image updated'
        
        return None, 'Image loaded. Click Auto-Generate to create initial mask.'
    
    return None, 'Invalid image format'


def get_current_mask() -> Optional[numpy.ndarray]:
    return CURRENT_MASK


def set_frame(frame: VisionFrame) -> None:
    global CURRENT_FRAME
    CURRENT_FRAME = frame


def set_mask(mask: numpy.ndarray) -> None:
    global CURRENT_MASK
    CURRENT_MASK = mask
