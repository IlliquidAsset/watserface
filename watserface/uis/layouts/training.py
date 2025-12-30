from typing import Optional, Any, Dict
import gradio

from watserface import state_manager
from watserface.uis.components import about, training_options, training_source, training_target, terminal
from watserface.training import core as training_core


def format_training_status(status_data: Any) -> str:
	"""Format training status for display"""
	if isinstance(status_data, list) and len(status_data) >= 2:
		message = status_data[0]
		telemetry = status_data[1] if isinstance(status_data[1], dict) else {}

		# Build formatted status
		output = f"{message}\n"

		# Add telemetry details if present
		if telemetry.get('status'):
			output += f"Status: {telemetry['status']}\n"
		if telemetry.get('frames_extracted'):
			output += f"Frames Extracted: {telemetry['frames_extracted']}\n"
		if telemetry.get('landmarks_saved'):
			output += f"Landmarks Saved: {telemetry['landmarks_saved']}\n"
		if telemetry.get('current_file'):
			output += f"File: {telemetry['current_file']}\n"
		if telemetry.get('epoch'):
			output += f"Epoch: {telemetry['epoch']}/{telemetry.get('total_epochs', '?')}\n"
		if telemetry.get('loss'):
			output += f"Loss: {telemetry['loss']:.4f}\n"
		if telemetry.get('model_path'):
			output += f"Model Path: {telemetry['model_path']}\n"

		return output.strip()

	return str(status_data)


def wrapped_start_identity_training(model_name: str, epochs: int, source_files: Any):
	"""Wrapper to format identity training output with throttling"""
	import time
	last_update = 0
	last_status = None

	for status in training_core.start_identity_training(model_name, epochs, source_files):
		current_time = time.time()
		formatted_status = format_training_status(status)

		# Only update UI every 0.5 seconds to avoid flickering
		if current_time - last_update >= 0.5 or formatted_status != last_status:
			last_update = current_time
			last_status = formatted_status
			yield formatted_status

	# Always yield the final status
	if last_status:
		yield last_status


def wrapped_start_occlusion_training(model_name: str, epochs: int, target_file: Any):
	"""Wrapper to format occlusion training output with throttling"""
	import time
	last_update = 0
	last_status = None

	for status in training_core.start_occlusion_training(model_name, epochs, target_file):
		current_time = time.time()
		formatted_status = format_training_status(status)

		# Only update UI every 0.5 seconds to avoid flickering
		if current_time - last_update >= 0.5 or formatted_status != last_status:
			last_update = current_time
			last_status = formatted_status
			yield formatted_status

	# Always yield the final status
	if last_status:
		yield last_status


def wrapped_stop_training():
	"""Wrapper to format stop training output"""
	result = training_core.stop_training()
	return format_training_status(result)


# Identity Training Components
IDENTITY_MODEL_NAME : Optional[gradio.Textbox] = None
IDENTITY_EPOCHS : Optional[gradio.Slider] = None
START_IDENTITY_BUTTON : Optional[gradio.Button] = None
STOP_IDENTITY_BUTTON : Optional[gradio.Button] = None
IDENTITY_STATUS : Optional[gradio.Textbox] = None

# Occlusion Training Components
OCCLUSION_MODEL_NAME : Optional[gradio.Textbox] = None
OCCLUSION_EPOCHS : Optional[gradio.Slider] = None
START_OCCLUSION_BUTTON : Optional[gradio.Button] = None
STOP_OCCLUSION_BUTTON : Optional[gradio.Button] = None
OCCLUSION_STATUS : Optional[gradio.Textbox] = None


def pre_check() -> bool:
	return True


def render() -> gradio.Blocks:
	global IDENTITY_MODEL_NAME, IDENTITY_EPOCHS, START_IDENTITY_BUTTON, STOP_IDENTITY_BUTTON, IDENTITY_STATUS
	global OCCLUSION_MODEL_NAME, OCCLUSION_EPOCHS, START_OCCLUSION_BUTTON, STOP_OCCLUSION_BUTTON, OCCLUSION_STATUS

	with gradio.Blocks() as layout:
		about.render()
		with gradio.Row():
			# --- Left Column: Train Identity (Source) ---
			with gradio.Column(scale=1):
				gradio.Markdown("### 👤 Train Identity Model (Source)")
				training_source.render()
				IDENTITY_MODEL_NAME = gradio.Textbox(label="Identity Model Name", placeholder="e.g. my_actor_v1")
				IDENTITY_EPOCHS = gradio.Slider(label="Epochs", minimum=1, maximum=1000, value=100, step=1)
				with gradio.Row():
					START_IDENTITY_BUTTON = gradio.Button("Start Identity Training", variant="primary")
					STOP_IDENTITY_BUTTON = gradio.Button("Stop", variant="stop")
				IDENTITY_STATUS = gradio.Textbox(label="Identity Training Status", value="Idle", interactive=False)

			# --- Right Column: Train Occlusion (Target) ---
			with gradio.Column(scale=1):
				gradio.Markdown("### 🎭 Train Occlusion Model (Target)")
				training_target.render()
				OCCLUSION_MODEL_NAME = gradio.Textbox(label="Occlusion Model Name", placeholder="e.g. corndog_scene_mask")
				OCCLUSION_EPOCHS = gradio.Slider(label="Epochs", minimum=1, maximum=200, value=50, step=1)
				with gradio.Row():
					START_OCCLUSION_BUTTON = gradio.Button("Start Occlusion Training", variant="primary")
					STOP_OCCLUSION_BUTTON = gradio.Button("Stop", variant="stop")
				OCCLUSION_STATUS = gradio.Textbox(label="Occlusion Training Status", value="Idle", interactive=False)

		with gradio.Row():
			terminal.render()
				
	return layout


def listen() -> None:
	training_source.listen()
	training_target.listen()
	terminal.listen()
	
	START_IDENTITY_BUTTON.click(
		wrapped_start_identity_training,
		inputs=[IDENTITY_MODEL_NAME, IDENTITY_EPOCHS, training_source.TRAINING_SOURCE_FILE],
		outputs=[IDENTITY_STATUS]
	)
	STOP_IDENTITY_BUTTON.click(wrapped_stop_training, outputs=[IDENTITY_STATUS])

	START_OCCLUSION_BUTTON.click(
		wrapped_start_occlusion_training,
		inputs=[OCCLUSION_MODEL_NAME, OCCLUSION_EPOCHS, training_target.TRAINING_TARGET_FILE],
		outputs=[OCCLUSION_STATUS]
	)
	STOP_OCCLUSION_BUTTON.click(wrapped_stop_training, outputs=[OCCLUSION_STATUS])


def run(ui : gradio.Blocks) -> None:
	ui.launch(inbrowser = state_manager.get_item('open_browser'), server_name = state_manager.get_item('server_name'), server_port = state_manager.get_item('server_port'), show_error = True)
