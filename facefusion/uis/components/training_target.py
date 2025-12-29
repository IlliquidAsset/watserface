from typing import Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.face_store import clear_reference_faces, clear_static_faces
from facefusion.filesystem import is_image, is_video
from facefusion.uis.core import register_ui_component
from facefusion.uis.types import ComponentOptions, File
from facefusion.vision import count_video_frame_total

TRAINING_TARGET_FILE : Optional[gradio.File] = None
TRAINING_TARGET_IMAGE : Optional[gradio.Image] = None
TRAINING_TARGET_VIDEO : Optional[gradio.Video] = None
TRAINING_TARGET_GALLERY : Optional[gradio.Gallery] = None
TRAINING_TARGET_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global TRAINING_TARGET_FILE
	global TRAINING_TARGET_IMAGE
	global TRAINING_TARGET_VIDEO
	global TRAINING_TARGET_GALLERY
	global TRAINING_TARGET_SLIDER

	is_training_target_image = is_image(state_manager.get_item('training_target_path'))
	is_training_target_video = is_video(state_manager.get_item('training_target_path'))
	TRAINING_TARGET_FILE = gradio.File(
		label = wording.get('uis.target_file'),
		file_types = [ 'image', 'video' ],
		value = state_manager.get_item('training_target_path') if is_training_target_image or is_training_target_video else None
	)
	training_target_image_options : ComponentOptions =\
	{
		'show_label': False,
		'visible': False
	}
	training_target_video_options : ComponentOptions =\
	{
		'show_label': False,
		'visible': False
	}
	training_target_gallery_options : ComponentOptions =\
	{
		'label': 'Target Frames',
		'object_fit': 'cover',
		'columns': 4,
		'rows': 2,
		'height': 'auto',
		'visible': False
	}
	training_target_slider_options : ComponentOptions =\
	{
		'label': 'Preview Frame',
		'minimum': 0,
		'step': 1,
		'value': 0,
		'visible': False
	}

	if is_training_target_image:
		training_target_image_options['value'] = TRAINING_TARGET_FILE.value.get('path') if isinstance(TRAINING_TARGET_FILE.value, dict) else TRAINING_TARGET_FILE.value
		training_target_image_options['visible'] = True
	if is_training_target_video:
		training_target_video_options['value'] = TRAINING_TARGET_FILE.value.get('path') if isinstance(TRAINING_TARGET_FILE.value, dict) else TRAINING_TARGET_FILE.value
		training_target_video_options['visible'] = True
		# Enable slider for video
		video_path = training_target_video_options['value']
		if video_path:
			frame_total = count_video_frame_total(video_path)
			training_target_slider_options['maximum'] = frame_total - 1
			training_target_slider_options['visible'] = True

	TRAINING_TARGET_IMAGE = gradio.Image(**training_target_image_options)
	TRAINING_TARGET_VIDEO = gradio.Video(**training_target_video_options)
	TRAINING_TARGET_GALLERY = gradio.Gallery(**training_target_gallery_options)
	TRAINING_TARGET_SLIDER = gradio.Slider(**training_target_slider_options)

	register_ui_component('training_target_image', TRAINING_TARGET_IMAGE)
	register_ui_component('training_target_video', TRAINING_TARGET_VIDEO)
	register_ui_component('training_target_gallery', TRAINING_TARGET_GALLERY)
	register_ui_component('training_target_slider', TRAINING_TARGET_SLIDER)


def listen() -> None:
	TRAINING_TARGET_FILE.change(update, inputs = TRAINING_TARGET_FILE, outputs = [ TRAINING_TARGET_IMAGE, TRAINING_TARGET_VIDEO, TRAINING_TARGET_GALLERY, TRAINING_TARGET_SLIDER ])

def update(file : File) -> Tuple[gradio.Image, gradio.Video, gradio.Gallery, gradio.Slider]:
	if file and is_image(file.name):
		state_manager.set_item('training_target_path', file.name)
		# Show image, hide video/gallery/slider
		return (
			gradio.Image(value = file.name, visible = True),
			gradio.Video(value = None, visible = False),
			gradio.Gallery(visible = False),
			gradio.Slider(visible = False)
		)

	if file and is_video(file.name):
		state_manager.set_item('training_target_path', file.name)
		# Show video and slider, hide image/gallery
		frame_total = count_video_frame_total(file.name)
		return (
			gradio.Image(value = None, visible = False),
			gradio.Video(value = file.name, visible = True),
			gradio.Gallery(visible = False),
			gradio.Slider(minimum = 0, maximum = frame_total - 1, value = 0, visible = True)
		)

	state_manager.clear_item('training_target_path')
	return (
		gradio.Image(value = None, visible = False),
		gradio.Video(value = None, visible = False),
		gradio.Gallery(visible = False),
		gradio.Slider(visible = False)
	)
