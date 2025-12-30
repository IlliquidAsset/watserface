from typing import Any, List, Optional, Tuple

import gradio

from watserface import state_manager, wording
from watserface.face_store import clear_reference_faces, clear_static_faces
from watserface.filesystem import is_image, is_video
from watserface.uis.core import register_ui_component
from watserface.uis.types import ComponentOptions, File
from watserface.vision import count_video_frame_total

TRAINING_SOURCE_FILE : Optional[gradio.File] = None
TRAINING_SOURCE_IMAGE : Optional[gradio.Image] = None
TRAINING_SOURCE_VIDEO : Optional[gradio.Video] = None
TRAINING_SOURCE_GALLERY : Optional[gradio.Gallery] = None
TRAINING_SOURCE_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global TRAINING_SOURCE_FILE
	global TRAINING_SOURCE_IMAGE
	global TRAINING_SOURCE_VIDEO
	global TRAINING_SOURCE_GALLERY
	global TRAINING_SOURCE_SLIDER

	is_training_source_image = is_image(state_manager.get_item('training_source_path'))
	is_training_source_video = is_video(state_manager.get_item('training_source_path'))
	TRAINING_SOURCE_FILE = gradio.File(
		label = wording.get('uis.source_file'),
		file_types = [ 'image', 'video' ],
		file_count = 'multiple',
		value = state_manager.get_item('training_source_path') if is_training_source_image or is_training_source_video else None
	)
	training_source_image_options : ComponentOptions =\
	{
		'show_label': False,
		'visible': False
	}
	training_source_video_options : ComponentOptions =\
	{
		'show_label': False,
		'visible': False
	}
	training_source_gallery_options : ComponentOptions =\
	{
		'label': 'Source Files',
		'object_fit': 'cover',
		'columns': 4,
		'rows': 2,
		'height': 'auto',
		'visible': False
	}
	training_source_slider_options : ComponentOptions =\
	{
		'label': 'Preview Frame',
		'minimum': 0,
		'step': 1,
		'value': 0,
		'visible': False
	}

	if is_training_source_image:
		training_source_image_options['value'] = TRAINING_SOURCE_FILE.value.get('path') if isinstance(TRAINING_SOURCE_FILE.value, dict) else TRAINING_SOURCE_FILE.value
		training_source_image_options['visible'] = True
	if is_training_source_video:
		training_source_video_options['value'] = TRAINING_SOURCE_FILE.value.get('path') if isinstance(TRAINING_SOURCE_FILE.value, dict) else TRAINING_SOURCE_FILE.value
		training_source_video_options['visible'] = True
		# Enable slider for video
		video_path = training_source_video_options['value']
		if video_path:
			frame_total = count_video_frame_total(video_path)
			training_source_slider_options['maximum'] = frame_total - 1
			training_source_slider_options['visible'] = True

	TRAINING_SOURCE_IMAGE = gradio.Image(**training_source_image_options)
	TRAINING_SOURCE_VIDEO = gradio.Video(**training_source_video_options)
	TRAINING_SOURCE_GALLERY = gradio.Gallery(**training_source_gallery_options)
	TRAINING_SOURCE_SLIDER = gradio.Slider(**training_source_slider_options)

	register_ui_component('training_source_image', TRAINING_SOURCE_IMAGE)
	register_ui_component('training_source_video', TRAINING_SOURCE_VIDEO)
	register_ui_component('training_source_gallery', TRAINING_SOURCE_GALLERY)
	register_ui_component('training_source_slider', TRAINING_SOURCE_SLIDER)


def listen() -> None:
	TRAINING_SOURCE_FILE.change(update, inputs = TRAINING_SOURCE_FILE, outputs = [ TRAINING_SOURCE_IMAGE, TRAINING_SOURCE_VIDEO, TRAINING_SOURCE_GALLERY, TRAINING_SOURCE_SLIDER ])

def update(files : Any) -> Tuple[gradio.Image, gradio.Video, gradio.Gallery, gradio.Slider]:
	# Handle multiple files or single file
	file = files[0] if isinstance(files, list) and files else files

	if file and is_image(file.name):
		state_manager.set_item('training_source_path', file.name)
		# Show image, hide video and slider, show gallery if multiple files
		gallery_visible = isinstance(files, list) and len(files) > 1
		gallery_value = [f.name for f in files] if gallery_visible else None
		return (
			gradio.Image(value = file.name, visible = True),
			gradio.Video(value = None, visible = False),
			gradio.Gallery(value = gallery_value, visible = gallery_visible),
			gradio.Slider(visible = False)
		)

	if file and is_video(file.name):
		state_manager.set_item('training_source_path', file.name)
		# Show video and slider, hide image
		frame_total = count_video_frame_total(file.name)
		return (
			gradio.Image(value = None, visible = False),
			gradio.Video(value = file.name, visible = True),
			gradio.Gallery(visible = False),
			gradio.Slider(minimum = 0, maximum = frame_total - 1, value = 0, visible = True)
		)

	state_manager.clear_item('training_source_path')
	return (
		gradio.Image(value = None, visible = False),
		gradio.Video(value = None, visible = False),
		gradio.Gallery(visible = False),
		gradio.Slider(visible = False)
	)
