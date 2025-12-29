from typing import Optional
import gradio

from facefusion import state_manager
from facefusion.uis.components import about, training_options, training_source, training_target, terminal
from facefusion.training import core as training_core

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
		training_core.start_identity_training,
		inputs=[IDENTITY_MODEL_NAME, IDENTITY_EPOCHS, training_source.TRAINING_SOURCE_FILE],
		outputs=[IDENTITY_STATUS]
	)
	STOP_IDENTITY_BUTTON.click(training_core.stop_training, outputs=[IDENTITY_STATUS])

	START_OCCLUSION_BUTTON.click(
		training_core.start_occlusion_training,
		inputs=[OCCLUSION_MODEL_NAME, OCCLUSION_EPOCHS, training_target.TRAINING_TARGET_FILE],
		outputs=[OCCLUSION_STATUS]
	)
	STOP_OCCLUSION_BUTTON.click(training_core.stop_training, outputs=[OCCLUSION_STATUS])


def run(ui : gradio.Blocks) -> None:
	ui.launch(favicon_path = 'facefusion.ico', inbrowser = state_manager.get_item('open_browser'), server_name = state_manager.get_item('server_name'), server_port = state_manager.get_item('server_port'), show_error = True)
