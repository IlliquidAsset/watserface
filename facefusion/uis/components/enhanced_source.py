from typing import List, Optional, Tuple
import gradio

from facefusion import state_manager, wording
from facefusion.common_helper import get_first
from facefusion.filesystem import filter_audio_paths, filter_image_paths, has_audio, has_image
from facefusion.uis.core import register_ui_component
from facefusion.uis.types import File

SOURCE_FILE: Optional[gradio.File] = None
SOURCE_AUDIO: Optional[gradio.Audio] = None
SOURCE_IMAGE: Optional[gradio.Image] = None
SOURCE_STATUS: Optional[gradio.Textbox] = None


def render() -> None:
    """Render enhanced source upload with better UX"""
    global SOURCE_FILE, SOURCE_AUDIO, SOURCE_IMAGE, SOURCE_STATUS

    has_source_audio = has_audio(state_manager.get_item('source_paths'))
    has_source_image = has_image(state_manager.get_item('source_paths'))
    
    # Enhanced file upload with better messaging
    with gradio.Column():
        gradio.Markdown(
            """
            ### 📸 Source Face
            Upload a clear photo of the person whose face you want to use.
            
            **💡 Tips for best results:**
            - Use a front-facing, well-lit photo
            - Person should be looking directly at camera
            - Avoid sunglasses, masks, or extreme angles
            - Higher resolution images work better
            """
        )
        
        SOURCE_FILE = gradio.File(
            label="📁 Drop your source photo here or click to browse",
            file_count='multiple',
            file_types=['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
            value=state_manager.get_item('source_paths') if has_source_audio or has_source_image else None,
            elem_id="enhanced_source_file",
            height=120
        )
        
        # Status indicator
        SOURCE_STATUS = gradio.Textbox(
            label="Status",
            value="👆 Upload a source image to get started",
            interactive=False,
            lines=1,
            elem_id="source_status"
        )
        
        # Preview containers
        source_file_names = [source_file_value.get('path') for source_file_value in SOURCE_FILE.value] if SOURCE_FILE.value else None
        source_audio_path = get_first(filter_audio_paths(source_file_names))
        source_image_path = get_first(filter_image_paths(source_file_names))
        
        with gradio.Column(visible=has_source_image or has_source_audio) as preview_container:
            gradio.Markdown("### 👁️ Preview")
            
            SOURCE_AUDIO = gradio.Audio(
                value=source_audio_path if has_source_audio else None,
                visible=has_source_audio,
                show_label=False,
                elem_id="source_audio_preview"
            )
            
            SOURCE_IMAGE = gradio.Image(
                value=source_image_path if has_source_image else None,
                visible=has_source_image,
                show_label=False,
                height=300,
                elem_id="source_image_preview"
            )

    # Register components
    register_ui_component('enhanced_source_file', SOURCE_FILE)
    register_ui_component('enhanced_source_audio', SOURCE_AUDIO)
    register_ui_component('enhanced_source_image', SOURCE_IMAGE)
    register_ui_component('enhanced_source_status', SOURCE_STATUS)
    register_ui_component('enhanced_source_preview_container', preview_container)


def listen() -> None:
    """Set up enhanced event listeners with better feedback"""
    if SOURCE_FILE and SOURCE_STATUS:
        SOURCE_FILE.change(
            update_with_feedback, 
            inputs=[SOURCE_FILE], 
            outputs=[SOURCE_AUDIO, SOURCE_IMAGE, SOURCE_STATUS]
        )


def update_with_feedback(files: List[File]) -> Tuple[gradio.Audio, gradio.Image, str]:
    """Update source with enhanced user feedback"""
    if not files:
        state_manager.clear_item('source_paths')
        return (
            gradio.Audio(value=None, visible=False),
            gradio.Image(value=None, visible=False),
            "👆 Upload a source image to get started"
        )
    
    file_names = [file.name for file in files]
    has_source_audio = has_audio(file_names)
    has_source_image = has_image(file_names)
    
    if has_source_image or has_source_audio:
        source_audio_path = get_first(filter_audio_paths(file_names))
        source_image_path = get_first(filter_image_paths(file_names))
        state_manager.set_item('source_paths', file_names)
        
        # Enhanced status messaging
        status_msg = "✅ Source loaded successfully! "
        if has_source_image:
            status_msg += f"📸 Image ready for face swapping."
        if has_source_audio:
            status_msg += f"🎵 Audio detected."
        
        return (
            gradio.Audio(value=source_audio_path, visible=has_source_audio),
            gradio.Image(value=source_image_path, visible=has_source_image),
            status_msg
        )
    else:
        # Error handling with clear message
        unsupported_files = [file.name.split('.')[-1] for file in files]
        status_msg = f"❌ Unsupported file format: {', '.join(unsupported_files)}. Please upload JPG, PNG, or other image files."
        
        return (
            gradio.Audio(value=None, visible=False),
            gradio.Image(value=None, visible=False),
            status_msg
        )