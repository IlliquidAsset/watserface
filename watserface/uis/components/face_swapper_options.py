from typing import List, Optional, Tuple
import os

import gradio

from watserface import logger, state_manager, wording
from watserface.common_helper import get_first
from watserface.filesystem import get_file_name, resolve_file_paths, resolve_relative_path
from watserface.processors import choices as processors_choices
from watserface.processors.core import load_processor_module
from watserface.processors.modules import face_swapper
from watserface.processors.types import FaceSwapperModel
from watserface.uis.core import get_ui_component, register_ui_component


def get_available_models() -> List[str]:
	"""Get list of models that are actually downloaded and available"""
	available_models = []

	# Check standard models
	model_set = face_swapper.create_static_model_set('full')
	for model_name, model_options in model_set.items():
		# Check if model file exists
		model_sources = model_options.get('sources', {})
		face_swapper_source = model_sources.get('face_swapper', {})
		model_path = face_swapper_source.get('path')

		if model_path and os.path.exists(model_path):
			available_models.append(model_name)

	# Always include trained LoRA models (filter out identity models)
	trained_model_file_paths = resolve_file_paths(resolve_relative_path('../.assets/models/trained'))
	if trained_model_file_paths:
		for model_file_path in trained_model_file_paths:
			model_name = get_file_name(model_file_path)
			# Only include LoRA models (with _lora suffix), exclude identity models
			if '_lora' in model_name and model_name not in available_models:
				available_models.append(model_name)

	return available_models

FACE_SWAPPER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_SWAPPER_REFRESH_BUTTON : Optional[gradio.Button] = None
FACE_SWAPPER_PIXEL_BOOST_DROPDOWN : Optional[gradio.Dropdown] = None


def render() -> None:
	global FACE_SWAPPER_MODEL_DROPDOWN
	global FACE_SWAPPER_REFRESH_BUTTON
	global FACE_SWAPPER_PIXEL_BOOST_DROPDOWN

	has_face_swapper = 'face_swapper' in state_manager.get_item('processors')

	# Only show models that are actually available
	available_models = get_available_models()
	current_model = state_manager.get_item('face_swapper_model')

	# If current model isn't available, select first available or None
	if current_model not in available_models:
		current_model = available_models[0] if available_models else None
		if current_model:
			state_manager.set_item('face_swapper_model', current_model)

	with gradio.Row():
		FACE_SWAPPER_MODEL_DROPDOWN = gradio.Dropdown(
			label = wording.get('uis.face_swapper_model_dropdown'),
			choices = available_models,
			value = current_model,
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
	"""Refresh the model choices to include newly downloaded or trained LoRA models"""
	face_swapper.create_static_model_set.cache_clear()

	# Get available models (only those that exist on disk)
	available_models = get_available_models()

	# Update the processors choices to include any new trained LoRA models
	trained_model_file_paths = resolve_file_paths(resolve_relative_path('../.assets/models/trained'))
	if trained_model_file_paths:
		for model_file_path in trained_model_file_paths:
			model_name = get_file_name(model_file_path)
			# Only add LoRA models to processor choices
			if '_lora' in model_name and model_name not in processors_choices.face_swapper_set:
				processors_choices.face_swapper_set[model_name] = [ '128x128', '256x256' ]

	return gradio.Dropdown(choices = available_models)


def update_face_swapper_model(face_swapper_model : FaceSwapperModel) -> Tuple[gradio.Dropdown, gradio.Dropdown]:
	face_swapper_module = load_processor_module('face_swapper')
	face_swapper_module.clear_inference_pool()
	state_manager.set_item('face_swapper_model', face_swapper_model)

	# Silence logger during UI update to avoid "model not found" errors during initialization
	logger.disable()
	pre_check_result = face_swapper_module.pre_check()
	logger.enable()

	if pre_check_result:
		face_swapper_pixel_boost_choices = processors_choices.face_swapper_set.get(state_manager.get_item('face_swapper_model'))
		state_manager.set_item('face_swapper_pixel_boost', get_first(face_swapper_pixel_boost_choices))
		return gradio.Dropdown(value = state_manager.get_item('face_swapper_model')), gradio.Dropdown(value = state_manager.get_item('face_swapper_pixel_boost'), choices = face_swapper_pixel_boost_choices)
	return gradio.Dropdown(), gradio.Dropdown()


def update_face_swapper_pixel_boost(face_swapper_pixel_boost : str) -> None:
	state_manager.set_item('face_swapper_pixel_boost', face_swapper_pixel_boost)
