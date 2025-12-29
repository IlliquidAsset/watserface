from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, wording
from facefusion.common_helper import get_first
from facefusion.filesystem import get_file_name, resolve_file_paths, resolve_relative_path
from facefusion.processors import choices as processors_choices
from facefusion.processors.core import load_processor_module
from facefusion.processors.modules import face_swapper
from facefusion.processors.types import FaceSwapperModel
from facefusion.uis.core import get_ui_component, register_ui_component

FACE_SWAPPER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SWAPPER_REFRESH_BUTTON : Optional[gradio.Button] = None
FACE_SWAPPER_PIXEL_BOOST_DROPDOWN : Optional[gradio.Dropdown] = None


def render() -> None:
	global FACE_SWAPPER_MODEL_DROPDOWN
	global FACE_SWAPPER_REFRESH_BUTTON
	global FACE_SWAPPER_PIXEL_BOOST_DROPDOWN

	has_face_swapper = 'face_swapper' in state_manager.get_item('processors')
	with gradio.Row():
		FACE_SWAPPER_MODEL_DROPDOWN = gradio.Dropdown(
			label = wording.get('uis.face_swapper_model_dropdown'),
			choices = processors_choices.face_swapper_models,
			value = state_manager.get_item('face_swapper_model'),
			visible = has_face_swapper,
			scale = 8
		)
		FACE_SWAPPER_REFRESH_BUTTON = gradio.Button(
			value = '🔄',
			variant = 'secondary',
			visible = has_face_swapper,
			scale = 1,
			elem_classes = [ 'ui_button_small' ]
		)
	FACE_SWAPPER_PIXEL_BOOST_DROPDOWN = gradio.Dropdown(
		label = wording.get('uis.face_swapper_pixel_boost_dropdown'),
		choices = processors_choices.face_swapper_set.get(state_manager.get_item('face_swapper_model')),
		value = state_manager.get_item('face_swapper_pixel_boost'),
		visible = has_face_swapper
	)
	register_ui_component('face_swapper_model_dropdown', FACE_SWAPPER_MODEL_DROPDOWN)
	register_ui_component('face_swapper_refresh_button', FACE_SWAPPER_REFRESH_BUTTON)
	register_ui_component('face_swapper_pixel_boost_dropdown', FACE_SWAPPER_PIXEL_BOOST_DROPDOWN)


def listen() -> None:
	FACE_SWAPPER_MODEL_DROPDOWN.change(update_face_swapper_model, inputs = FACE_SWAPPER_MODEL_DROPDOWN, outputs = [ FACE_SWAPPER_MODEL_DROPDOWN, FACE_SWAPPER_PIXEL_BOOST_DROPDOWN ])
	FACE_SWAPPER_REFRESH_BUTTON.click(update_face_swapper_choices, outputs = [ FACE_SWAPPER_MODEL_DROPDOWN ])
	FACE_SWAPPER_PIXEL_BOOST_DROPDOWN.change(update_face_swapper_pixel_boost, inputs = FACE_SWAPPER_PIXEL_BOOST_DROPDOWN)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ FACE_SWAPPER_MODEL_DROPDOWN, FACE_SWAPPER_REFRESH_BUTTON, FACE_SWAPPER_PIXEL_BOOST_DROPDOWN ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Button, gradio.Dropdown]:
	has_face_swapper = 'face_swapper' in processors
	return gradio.Dropdown(visible = has_face_swapper), gradio.Button(visible = has_face_swapper), gradio.Dropdown(visible = has_face_swapper)


def update_face_swapper_choices() -> gradio.Dropdown:
	face_swapper.create_static_model_set.cache_clear()
	processors_choices.face_swapper_models = list(processors_choices.face_swapper_set.keys())
	
	trained_model_file_paths = resolve_file_paths(resolve_relative_path('../.assets/models/trained'))
	if trained_model_file_paths:
		for model_file_path in trained_model_file_paths:
			model_name = get_file_name(model_file_path)
			if model_name not in processors_choices.face_swapper_models:
				processors_choices.face_swapper_models.append(model_name)
				processors_choices.face_swapper_set[model_name] = [ '256x256', '512x512' ]

	return gradio.Dropdown(choices = processors_choices.face_swapper_models)


def update_face_swapper_model(face_swapper_model : FaceSwapperModel) -> Tuple[gradio.Dropdown, gradio.Dropdown]:
	face_swapper_module = load_processor_module('face_swapper')
	face_swapper_module.clear_inference_pool()
	state_manager.set_item('face_swapper_model', face_swapper_model)

	if face_swapper_module.pre_check():
		face_swapper_pixel_boost_choices = processors_choices.face_swapper_set.get(state_manager.get_item('face_swapper_model'))
		state_manager.set_item('face_swapper_pixel_boost', get_first(face_swapper_pixel_boost_choices))
		return gradio.Dropdown(value = state_manager.get_item('face_swapper_model')), gradio.Dropdown(value = state_manager.get_item('face_swapper_pixel_boost'), choices = face_swapper_pixel_boost_choices)
	return gradio.Dropdown(), gradio.Dropdown()


def update_face_swapper_pixel_boost(face_swapper_pixel_boost : str) -> None:
	state_manager.set_item('face_swapper_pixel_boost', face_swapper_pixel_boost)
