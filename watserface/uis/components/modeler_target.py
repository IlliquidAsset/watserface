"""
Modeler Target Component - Target Video/Image Uploader
"""
from typing import Optional, Tuple, List
import gradio

from watserface import state_manager, wording, logger
from watserface.common_helper import get_first
from watserface.filesystem import has_image, has_video, is_video
from watserface.uis.core import register_ui_component
from watserface.uis.types import File
from watserface.vision import count_video_frame_total


MODELER_TARGET_FILE: Optional[gradio.File] = None
MODELER_TARGET_VIDEO: Optional[gradio.Video] = None
MODELER_TARGET_IMAGE: Optional[gradio.Image] = None
MODELER_TARGET_STATUS: Optional[gradio.Textbox] = None
MODELER_TARGET_INFO: Optional[gradio.Markdown] = None


def render() -> None:
	"""Render target uploader for Modeler tab"""
	global MODELER_TARGET_FILE, MODELER_TARGET_VIDEO, MODELER_TARGET_IMAGE, MODELER_TARGET_STATUS, MODELER_TARGET_INFO

	has_target_video = has_video(state_manager.get_item('modeler_target_path'))
	has_target_image = has_image(state_manager.get_item('modeler_target_path'))

	with gradio.Column():
		gradio.Markdown(
			"""
			### 🎯 Target Material
			Upload the target video or image to train against.

			**💡 Tips:**
			- **Video**: Best for creating LoRA models (more training data)
			- **Image**: Quick training, but less robust
			- Target should contain the face/scene you want to learn
			- Higher quality = better results
			"""
		)

		MODELER_TARGET_FILE = gradio.File(
			label="📁 Drop target video/image here or click to browse",
			file_count='single',
			file_types=['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.mp4', '.mov', '.avi', '.mkv'],
			value=state_manager.get_item('modeler_target_path'),
			elem_id="modeler_target_file",
			height=120
		)

		# Status display
		MODELER_TARGET_STATUS = gradio.Textbox(
			label="Status",
			value="👆 Upload target video or image to continue",
			interactive=False,
			lines=2,
			elem_id="modeler_target_status"
		)

		# Target information display
		MODELER_TARGET_INFO = gradio.Markdown(
			"",
			visible=False,
			elem_id="modeler_target_info"
		)

		# Preview containers
		target_file_path = state_manager.get_item('modeler_target_path')

		with gradio.Column(visible=has_target_video or has_target_image) as preview_container:
			gradio.Markdown("### 👁️ Preview")

			MODELER_TARGET_VIDEO = gradio.Video(
				value=target_file_path if has_target_video else None,
				visible=has_target_video,
				show_label=False,
				elem_id="modeler_target_video_preview",
				height=300
			)

			MODELER_TARGET_IMAGE = gradio.Image(
				value=target_file_path if has_target_image else None,
				visible=has_target_image,
				show_label=False,
				elem_id="modeler_target_image_preview",
				height=300
			)

	# Register components
	register_ui_component('modeler_target_file', MODELER_TARGET_FILE)
	register_ui_component('modeler_target_video', MODELER_TARGET_VIDEO)
	register_ui_component('modeler_target_image', MODELER_TARGET_IMAGE)
	register_ui_component('modeler_target_status', MODELER_TARGET_STATUS)
	register_ui_component('modeler_target_info', MODELER_TARGET_INFO)
	register_ui_component('modeler_target_preview_container', preview_container)


def listen() -> None:
	"""Set up event listeners"""
	if MODELER_TARGET_FILE and MODELER_TARGET_STATUS:
		MODELER_TARGET_FILE.change(
			update_target_file,
			inputs=[MODELER_TARGET_FILE],
			outputs=[MODELER_TARGET_VIDEO, MODELER_TARGET_IMAGE, MODELER_TARGET_STATUS, MODELER_TARGET_INFO]
		)


def update_target_file(file: File = None) -> Tuple[gradio.Video, gradio.Image, str, str]:
	"""Update state when target file is uploaded"""
	if not file:
		state_manager.clear_item('modeler_target_path')
		return (
			gradio.Video(value=None, visible=False),
			gradio.Image(value=None, visible=False),
			"👆 Upload target video or image to continue",
			""
		)

	file_path = file.name if hasattr(file, 'name') else file
	state_manager.set_item('modeler_target_path', file_path)

	# Determine if video or image
	if is_video(file_path):
		try:
			frame_count = count_video_frame_total(file_path)
			status_msg = f"✅ Video loaded: {frame_count} frames"

			target_info = f"""
### 🎬 Target Video Loaded

- **Frames**: {frame_count}
- **Recommended Epochs**: {min(100, max(50, frame_count // 10))}
- **Estimated Training Time**: ~{frame_count // 10} minutes (on GPU)

**✅ Ready for paired training**
"""
		except Exception as e:
			logger.error(f"Error loading video: {e}", __name__)
			status_msg = "⚠️ Video loaded, but couldn't read metadata"
			target_info = "**✅ Ready for paired training**"

		return (
			gradio.Video(value=file_path, visible=True),
			gradio.Image(value=None, visible=False),
			status_msg,
			target_info
		)
	else:
		status_msg = "✅ Image loaded"
		target_info = """
### 🖼️ Target Image Loaded

- **Type**: Single frame
- **Recommended Epochs**: 20-50
- **Estimated Training Time**: ~5 minutes (on GPU)

**⚠️ Note**: Video targets provide more robust training data

**✅ Ready for paired training**
"""
		return (
			gradio.Video(value=None, visible=False),
			gradio.Image(value=file_path, visible=True),
			status_msg,
			target_info
		)
