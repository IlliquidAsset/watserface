"""
Modeler Tab Layout - Paired Identity Training
"""
from typing import Optional, Any
import gradio

from watserface import state_manager, logger
from watserface.uis.components import about, footer, modeler_source, modeler_target, modeler_options, terminal


# Training Controls
START_MODELER_BUTTON: Optional[gradio.Button] = None
STOP_MODELER_BUTTON: Optional[gradio.Button] = None
MODELER_STATUS: Optional[gradio.Textbox] = None


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
		if telemetry.get('epoch'):
			output += f"Epoch: {telemetry['epoch']}/{telemetry.get('total_epochs', '?')}\n"
		if telemetry.get('loss'):
			output += f"Loss: {telemetry['loss']:.4f}\n"
		if telemetry.get('model_path'):
			output += f"Model Path: {telemetry['model_path']}\n"

		return output.strip()

	return str(status_data)


def wrapped_start_lora_training(
	source_profile_id: str,
	target_file: Any,
	model_name: str,
	epochs: int,
	learning_rate: float,
	lora_rank: int,
	batch_size: int
):
	"""Wrapper to format LoRA training output with throttling"""
	import time
	import os
	import shutil
	from watserface.training.dataset_extractor import extract_training_dataset
	from watserface.training.landmark_smoother import apply_smoothing_to_dataset
	from watserface.training.train_lora import train_lora_model

	# Validation
	if not source_profile_id:
		yield "❌ Error: Please select a source identity profile"
		return

	if not target_file:
		yield "❌ Error: Please upload a target video or image"
		return

	if not model_name:
		yield "❌ Error: Please enter a model name"
		return

	# Get target file path
	target_path = target_file.name if hasattr(target_file, 'name') else target_file

	# Create dataset directory
	dataset_path = os.path.join(state_manager.get_item('jobs_path'), 'training_dataset_lora')
	if not os.path.exists(dataset_path):
		os.makedirs(dataset_path, exist_ok=True)

	# Check if frames already extracted
	existing_frames = len([f for f in os.listdir(dataset_path) if f.endswith('.png')]) if os.path.exists(dataset_path) else 0

	if existing_frames > 0:
		yield f"📂 Using {existing_frames} existing frames. Skipping extraction..."
	else:
		# Step 1: Extract frames from target
		yield "📹 Extracting frames from target video..."
		for stats in extract_training_dataset(
			source_paths=[target_path],
			output_dir=dataset_path,
			frame_interval=2,
			max_frames=1000
		):
			yield f"Extracting... {stats.get('frames_extracted', 0)} frames"

		# Step 2: Apply smoothing
		yield "🎨 Applying landmark smoothing..."
		apply_smoothing_to_dataset(dataset_path)

	# Step 3: Train LoRA model
	yield "🚀 Starting LoRA training..."

	last_update = 0
	last_status = None

	for status_data in train_lora_model(
		dataset_dir=dataset_path,
		source_profile_id=source_profile_id,
		model_name=model_name,
		epochs=int(epochs),
		batch_size=int(batch_size),
		learning_rate=float(learning_rate),
		lora_rank=int(lora_rank),
		save_interval=max(10, int(epochs) // 5)
	):
		current_time = time.time()
		formatted_status = format_training_status(status_data)

		# Throttle UI updates to avoid flickering (every 0.5s)
		if current_time - last_update >= 0.5 or formatted_status != last_status:
			last_update = current_time
			last_status = formatted_status
			yield formatted_status

	# Always yield the final status
	if last_status:
		yield last_status


def wrapped_stop_training():
	"""Wrapper to format stop training output"""
	# TODO: Implement training stop mechanism
	return "🚧 Training stop not yet implemented"


def pre_check() -> bool:
	"""Pre-check for modeler tab"""
	return True


def render() -> gradio.Blocks:
	"""Render the Modeler tab layout"""
	global START_MODELER_BUTTON, STOP_MODELER_BUTTON, MODELER_STATUS

	with gradio.Blocks() as layout:
		about.render()

		gradio.Markdown("## 🎯 Step 3: LoRA Paired Training (Modeler)")
		gradio.Markdown(
			"""
			Train a custom LoRA model that maps a **source identity** to a **target scene/person**.
			This creates a specialized model for highly accurate face swapping in specific contexts.

			**How it works:**
			1. Select a trained identity profile (from Training tab)
			2. Upload target video/image to train against
			3. Configure LoRA training parameters
			4. Start training to create a custom model
			5. Use the trained model in the Swap tab for superior results

			**Benefits:**
			- 🎯 Superior accuracy for specific source→target pairs
			- 🚀 Smaller model size (LoRA adapters)
			- 💾 Faster inference than full models
			- 🔄 Can combine multiple LoRA models
			"""
		)

		with gradio.Row():
			# Left Column: Source Identity
			with gradio.Column(scale=1):
				with gradio.Accordion("👤 Source Identity", open=True):
					modeler_source.render()

			# Right Column: Target Material
			with gradio.Column(scale=1):
				with gradio.Accordion("🎯 Target Material", open=True):
					modeler_target.render()

		# Training Configuration
		with gradio.Accordion("⚙️ Training Configuration", open=True):
			modeler_options.render()

		# Training Controls
		with gradio.Row():
			START_MODELER_BUTTON = gradio.Button(
				"🚀 Start LoRA Training",
				variant="primary",
				size="lg"
			)
			STOP_MODELER_BUTTON = gradio.Button(
				"⏹️ Stop",
				variant="stop",
				size="lg"
			)

		# Status Display
		MODELER_STATUS = gradio.Textbox(
			label="Training Status",
			value="Idle - Configure settings and click 'Start LoRA Training'",
			interactive=False,
			lines=8,
			elem_id="modeler_training_status"
		)

		# Terminal for debugging
		with gradio.Row():
			terminal.render()

		footer.render()

	return layout


def listen() -> None:
	"""Set up event listeners"""
	modeler_source.listen()
	modeler_target.listen()
	modeler_options.listen()
	terminal.listen()

	# Wire up training buttons
	START_MODELER_BUTTON.click(
		wrapped_start_lora_training,
		inputs=[
			modeler_source.MODELER_SOURCE_PROFILE_DROPDOWN,
			modeler_target.MODELER_TARGET_FILE,
			modeler_options.MODELER_MODEL_NAME,
			modeler_options.MODELER_EPOCHS,
			modeler_options.MODELER_LEARNING_RATE,
			modeler_options.MODELER_LORA_RANK,
			modeler_options.MODELER_BATCH_SIZE
		],
		outputs=[MODELER_STATUS]
	)

	STOP_MODELER_BUTTON.click(
		wrapped_stop_training,
		outputs=[MODELER_STATUS]
	)


def run(ui: gradio.Blocks) -> None:
	"""Run the modeler UI"""
	ui.launch(
		inbrowser=state_manager.get_item('open_browser'),
		server_name=state_manager.get_item('server_name'),
		server_port=state_manager.get_item('server_port'),
		show_error=True
	)
