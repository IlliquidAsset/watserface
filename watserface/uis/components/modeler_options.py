"""
Modeler Options Component - Training Configuration
"""
from typing import Optional
import gradio

from watserface import state_manager, wording
from watserface.uis.core import register_ui_component


MODELER_MODEL_NAME: Optional[gradio.Textbox] = None
MODELER_EPOCHS: Optional[gradio.Slider] = None
MODELER_LEARNING_RATE: Optional[gradio.Slider] = None
MODELER_LORA_RANK: Optional[gradio.Slider] = None
MODELER_BATCH_SIZE: Optional[gradio.Dropdown] = None


def render() -> None:
	"""Render training configuration options"""
	global MODELER_MODEL_NAME, MODELER_EPOCHS, MODELER_LEARNING_RATE, MODELER_LORA_RANK, MODELER_BATCH_SIZE

	with gradio.Column():
		gradio.Markdown("### ⚙️ Training Configuration")

		MODELER_MODEL_NAME = gradio.Textbox(
			label="Model Name",
			placeholder="e.g. actor_to_corndog_scene",
			value="",
			info="Unique name for this paired model",
			elem_id="modeler_model_name"
		)

		with gradio.Row():
			MODELER_EPOCHS = gradio.Slider(
				label="Epochs",
				minimum=1,
				maximum=500,
				value=100,
				step=1,
				info="More epochs = better quality, but longer training time",
				elem_id="modeler_epochs"
			)

			MODELER_LORA_RANK = gradio.Slider(
				label="LoRA Rank",
				minimum=4,
				maximum=128,
				value=16,
				step=4,
				info="Higher rank = more capacity, but larger model size",
				elem_id="modeler_lora_rank"
			)

		with gradio.Row():
			MODELER_LEARNING_RATE = gradio.Slider(
				label="Learning Rate",
				minimum=0.00001,
				maximum=0.001,
				value=0.0001,
				step=0.00001,
				info="Lower = more stable, higher = faster learning",
				elem_id="modeler_learning_rate"
			)

			MODELER_BATCH_SIZE = gradio.Dropdown(
				label="Batch Size",
				choices=[1, 2, 4, 8],
				value=4,
				info="Higher = faster, but requires more GPU memory",
				elem_id="modeler_batch_size"
			)

		gradio.Markdown(
			"""
			**💡 Recommended Settings:**
			- **Quick Test**: 20 epochs, rank 8, batch 2
			- **Balanced**: 100 epochs, rank 16, batch 4 (default)
			- **High Quality**: 200 epochs, rank 32, batch 4
			"""
		)

	# Register components
	register_ui_component('modeler_model_name', MODELER_MODEL_NAME)
	register_ui_component('modeler_epochs', MODELER_EPOCHS)
	register_ui_component('modeler_learning_rate', MODELER_LEARNING_RATE)
	register_ui_component('modeler_lora_rank', MODELER_LORA_RANK)
	register_ui_component('modeler_batch_size', MODELER_BATCH_SIZE)


def listen() -> None:
	"""Set up event listeners"""
	# Update state when options change
	if MODELER_MODEL_NAME:
		MODELER_MODEL_NAME.change(
			lambda x: state_manager.set_item('modeler_model_name', x),
			inputs=[MODELER_MODEL_NAME]
		)

	if MODELER_EPOCHS:
		MODELER_EPOCHS.change(
			lambda x: state_manager.set_item('modeler_epochs', x),
			inputs=[MODELER_EPOCHS]
		)

	if MODELER_LEARNING_RATE:
		MODELER_LEARNING_RATE.change(
			lambda x: state_manager.set_item('lora_learning_rate', x),
			inputs=[MODELER_LEARNING_RATE]
		)

	if MODELER_LORA_RANK:
		MODELER_LORA_RANK.change(
			lambda x: state_manager.set_item('lora_rank', x),
			inputs=[MODELER_LORA_RANK]
		)

	if MODELER_BATCH_SIZE:
		MODELER_BATCH_SIZE.change(
			lambda x: state_manager.set_item('modeler_batch_size', x),
			inputs=[MODELER_BATCH_SIZE]
		)
