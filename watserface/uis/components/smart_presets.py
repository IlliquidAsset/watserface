from typing import Optional, List
import gradio
from watserface import state_manager
from watserface.uis.core import register_ui_component, get_ui_component

SMART_PRESET_RADIO: Optional[gradio.Radio] = None

PRESETS = {
    'Fast': {
        'processors': ['face_swapper'],
        'face_swapper_model': 'inswapper_128',
        'face_enhancer_model': 'gfpgan_1.4', # Not used but good to reset
        'output_video_resolution': '720p',
        'execution_thread_count': 4,
        'execution_queue_count': 1
    },
    'Balanced': {
        'processors': ['face_swapper', 'face_enhancer'],
        'face_swapper_model': 'inswapper_128',
        'face_enhancer_model': 'gfpgan_1.4',
        'output_video_resolution': '1080p',
        'execution_thread_count': 4,
        'execution_queue_count': 1
    },
    'Quality': {
        'processors': ['face_swapper', 'face_enhancer'],
        'face_swapper_model': 'ghost_3_256',
        'face_enhancer_model': 'codeformer',
        'output_video_resolution': None, # Original
        'execution_thread_count': 2,
        'execution_queue_count': 1
    }
}

def render() -> None:
    global SMART_PRESET_RADIO
    SMART_PRESET_RADIO = gradio.Radio(
        label="Smart Presets",
        choices=list(PRESETS.keys()),
        value="Balanced",
        type="value"
    )
    register_ui_component('smart_preset_radio', SMART_PRESET_RADIO)

def listen() -> None:
    SMART_PRESET_RADIO.change(
        apply_preset,
        inputs=SMART_PRESET_RADIO,
        outputs=[
            get_ui_component('processors_checkbox_group'),
            get_ui_component('face_swapper_model_dropdown'),
            get_ui_component('face_enhancer_model_dropdown'),
            get_ui_component('output_video_resolution_dropdown'),
        ]
    )

def apply_preset(preset_name: str):
    if preset_name not in PRESETS:
        return (
            gradio.update(),
            gradio.update(),
            gradio.update(),
            gradio.update()
        )

    settings = PRESETS[preset_name]

    # Update State Manager
    state_manager.set_item('processors', settings['processors'])
    state_manager.set_item('face_swapper_model', settings['face_swapper_model'])
    state_manager.set_item('face_enhancer_model', settings['face_enhancer_model'])

    # Only update resolution if a target file is loaded
    from watserface.filesystem import is_video
    target_path = state_manager.get_item('target_path')

    if settings['output_video_resolution'] and is_video(target_path):
        state_manager.set_item('output_video_resolution', settings['output_video_resolution'])
        res_value = settings['output_video_resolution']
    else:
        res_value = None

    # Return updates for UI components
    return (
        gradio.update(value=settings['processors']),
        gradio.update(value=settings['face_swapper_model']),
        gradio.update(value=settings['face_enhancer_model']),
        gradio.update(value=res_value) if res_value else gradio.update()
    )
