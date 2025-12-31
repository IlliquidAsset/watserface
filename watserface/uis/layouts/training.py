from typing import Optional, Any
import gradio
import pandas as pd

from watserface import state_manager
from watserface.uis.components import about, footer, training_source, training_target, terminal
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

		# Progress information
		if telemetry.get('overall_progress'):
			output += f"Overall Progress: {telemetry['overall_progress']}\n"

		if telemetry.get('epoch_progress'):
			output += f"Epoch Progress: {telemetry['epoch_progress']}\n"

		# Extraction phase
		if telemetry.get('frames_extracted'):
			output += f"Frames Extracted: {telemetry['frames_extracted']}\n"
		if telemetry.get('landmarks_saved'):
			output += f"Landmarks Saved: {telemetry['landmarks_saved']}\n"
		if telemetry.get('current_file'):
			output += f"File: {telemetry['current_file']}\n"

		# Training phase
		if telemetry.get('epoch'):
			output += f"Epoch: {telemetry['epoch']}/{telemetry.get('total_epochs', '?')}\n"

		if telemetry.get('batch') and telemetry.get('total_batches'):
			output += f"Batch: {telemetry['batch']}/{telemetry['total_batches']}\n"

		if telemetry.get('loss') or telemetry.get('current_loss'):
			loss_val = telemetry.get('loss', telemetry.get('current_loss'))
			output += f"Loss: {loss_val}\n"

		if telemetry.get('epoch_time'):
			output += f"Epoch Time: {telemetry['epoch_time']}\n"

		if telemetry.get('eta'):
			output += f"ETA: {telemetry['eta']}\n"

		if telemetry.get('device'):
			output += f"Device: {telemetry['device']}\n"

		# Final results
		if telemetry.get('model_path'):
			output += f"Model Path: {telemetry['model_path']}\n"

		if telemetry.get('error'):
			output += f"❌ Error: {telemetry['error']}\n"

		return output.strip()

	return str(status_data)


def wrapped_start_identity_training(
	model_name: str,
	epochs: int,
	source_type: str,
	face_set_id: Optional[str] = None,
	source_files: Any = None,
	save_as_face_set: bool = False,
	face_set_name: Optional[str] = None,
	face_set_description: Optional[str] = None
):
	"""Wrapper to format identity training output with throttling and telemetry"""
	import time
	last_update = 0
	last_status = None
	loss_history = []

	# Determine parameters based on source type
	if source_type == "Face Set":
		training_kwargs = {
			'model_name': model_name,
			'epochs': epochs,
			'face_set_id': face_set_id
		}
	else:  # "Upload Files"
		training_kwargs = {
			'model_name': model_name,
			'epochs': epochs,
			'source_files': source_files,
			'save_as_face_set': save_as_face_set,
			'new_face_set_name': face_set_name if save_as_face_set else None
		}

	for status_data in training_core.start_identity_training(**training_kwargs):
		current_time = time.time()
		formatted_status = format_training_status(status_data)

		# Collect Telemetry
		plot_update = None
		if isinstance(status_data, list) and len(status_data) >= 2 and isinstance(status_data[1], dict):
			telemetry = status_data[1]
			if 'loss' in telemetry and 'epoch' in telemetry:
				try:
					# Append to history
					loss_history.append({
						'epoch': int(telemetry['epoch']),
						'loss': float(telemetry['loss'])
					})
					# Create DataFrame for LinePlot
					plot_update = pd.DataFrame(loss_history)
				except (ValueError, TypeError):
					pass

		# Only update UI every 0.5 seconds to avoid flickering
		if current_time - last_update >= 0.5 or formatted_status != last_status:
			last_update = current_time
			last_status = formatted_status
			yield formatted_status, plot_update

	# Always yield the final status
	if last_status:
		yield last_status, (pd.DataFrame(loss_history) if loss_history else None)


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


# Telemetry Components
LOSS_PLOT : Optional[gradio.LinePlot] = None


def render() -> gradio.Blocks:
	global IDENTITY_MODEL_NAME, IDENTITY_EPOCHS, START_IDENTITY_BUTTON, STOP_IDENTITY_BUTTON, IDENTITY_STATUS
	global OCCLUSION_MODEL_NAME, OCCLUSION_EPOCHS, START_OCCLUSION_BUTTON, STOP_OCCLUSION_BUTTON, OCCLUSION_STATUS
	global LOSS_PLOT

	with gradio.Blocks() as layout:
		about.render()
		with gradio.Row():
			# --- Left Column: Controls & Progress ---
			with gradio.Column(scale=1):
				with gradio.Tabs():
					with gradio.Tab("Identity Training"):
						gradio.Markdown("### 👤 Train Identity Model")
						training_source.render()
						IDENTITY_MODEL_NAME = gradio.Textbox(label="Identity Model Name", placeholder="e.g. my_actor_v1")
						IDENTITY_EPOCHS = gradio.Slider(label="Epochs", minimum=1, maximum=1000, value=100, step=1)
						with gradio.Row():
							START_IDENTITY_BUTTON = gradio.Button("Start Identity Training", variant="primary")
							STOP_IDENTITY_BUTTON = gradio.Button("Stop", variant="stop")
						IDENTITY_STATUS = gradio.Textbox(label="Identity Status", value="Idle", interactive=False, lines=5)

					with gradio.Tab("Occlusion Training"):
						gradio.Markdown("### 🎭 Train Occlusion Model")
						training_target.render()
						OCCLUSION_MODEL_NAME = gradio.Textbox(label="Occlusion Model Name", placeholder="e.g. corndog_scene_mask")
						OCCLUSION_EPOCHS = gradio.Slider(label="Epochs", minimum=1, maximum=200, value=50, step=1)
						with gradio.Row():
							START_OCCLUSION_BUTTON = gradio.Button("Start Occlusion Training", variant="primary")
							STOP_OCCLUSION_BUTTON = gradio.Button("Stop", variant="stop")
						OCCLUSION_STATUS = gradio.Textbox(label="Occlusion Status", value="Idle", interactive=False, lines=5)

			# --- Right Column: Real-time Charts ---
			with gradio.Column(scale=1):
				gradio.Markdown("### 📈 Telemetry")
				LOSS_PLOT = gradio.LinePlot(
					label="Training Loss",
					x="epoch",
					y="loss",
					title="Loss over Time",
					width=400,
					height=300,
					tooltip=["epoch", "loss"],
					overlay_point=True
				)
				# Placeholder for other metrics if needed

		with gradio.Row():
			terminal.render()

		footer.render()

	return layout


def listen() -> None:
	training_source.listen()
	training_target.listen()
	terminal.listen()

	START_IDENTITY_BUTTON.click(
		wrapped_start_identity_training,
		inputs=[
			IDENTITY_MODEL_NAME,
			IDENTITY_EPOCHS,
			training_source.SOURCE_TYPE_RADIO,
			training_source.FACE_SET_DROPDOWN,
			training_source.TRAINING_SOURCE_FILE,
			training_source.SAVE_AS_FACE_SET_CHECKBOX,
			training_source.FACE_SET_NAME,
			training_source.FACE_SET_DESCRIPTION
		],
		outputs=[IDENTITY_STATUS, LOSS_PLOT]
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
