"""
Modeler Tab Layout - Paired Identity Training
"""
from typing import Optional, Any
import gradio

from watserface import state_manager, logger
from watserface.uis.components import about, footer, modeler_source, modeler_target, modeler_options, terminal, lora_loader


# Training Controls
START_MODELER_BUTTON: Optional[gradio.Button] = None
STOP_MODELER_BUTTON: Optional[gradio.Button] = None
MODELER_STATUS: Optional[gradio.Textbox] = None
MODELER_LOSS_PLOT: Optional[gradio.LinePlot] = None


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
	lora_mode: str,
	existing_lora_name: Optional[str],
	source_profile_id: str,
	target_file: Any,
	model_name: str,
	epochs: int,
	learning_rate: float,
	lora_rank: int,
	batch_size: int
):
	"""Wrapper to format LoRA training output with throttling and graphs"""
	import time
	import os
	import shutil
	import pandas as pd
	from watserface.training.dataset_extractor import extract_training_dataset
	from watserface.training.landmark_smoother import apply_smoothing_to_dataset
	from watserface.training.train_lora import train_lora_model

	# Handle "Continue Training" mode
	if lora_mode == "Continue Training":
		if not existing_lora_name:
			yield "❌ Error: Please select an existing LoRA model to continue training", None
			return
		model_name = existing_lora_name  # Use existing LoRA name

	# Validation
	if not source_profile_id:
		yield "❌ Error: Please select a source identity profile", None
		return

	if not target_file and lora_mode == "New LoRA":
		yield "❌ Error: Please upload a target video or image", None
		return

	if not model_name:
		yield "❌ Error: Please enter a model name", None
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
		yield f"📂 Using {existing_frames} existing frames. Skipping extraction...", None
	else:
		# Step 1: Extract frames from target
		yield "📹 Extracting frames from target video...", None
		for stats in extract_training_dataset(
			source_paths=[target_path],
			output_dir=dataset_path,
			frame_interval=2,
			max_frames=1000
		):
			yield f"Extracting... {stats.get('frames_extracted', 0)} frames", None

		# Step 2: Apply smoothing
		yield "🎨 Applying landmark smoothing...", None
		apply_smoothing_to_dataset(dataset_path)

	# Step 3: Train LoRA model
	yield "🚀 Starting LoRA training...", None

	last_update = 0
	last_status = None
	loss_history = []

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

		# Parse status data
		message = status_data[0] if isinstance(status_data, (list, tuple)) else str(status_data)
		telemetry = status_data[1] if isinstance(status_data, (list, tuple)) and len(status_data) > 1 else {}

		# Format status message
		if telemetry:
			epoch = telemetry.get('epoch', '?')
			total_epochs = telemetry.get('total_epochs', '?')
			loss = telemetry.get('loss', 'N/A')
			eta = telemetry.get('eta', 'N/A')
			device = telemetry.get('device', 'N/A')
			rank = telemetry.get('lora_rank', '?')
			trainable = telemetry.get('trainable_params', '?')

			formatted_status = f"""📊 LoRA Training Progress

Epoch: {epoch}/{total_epochs}
Loss: {loss}
ETA: {eta}
Device: {device}
LoRA Rank: {rank}
Trainable Parameters: {trainable:,}

{message}"""

			# Track loss for graphing
			if loss != 'N/A':
				try:
					loss_history.append({'epoch': int(epoch), 'loss': float(loss)})
					plot_update = pd.DataFrame(loss_history)
				except:
					plot_update = None
			else:
				plot_update = None
		else:
			formatted_status = message
			plot_update = None

		# Throttle UI updates to avoid flickering (every 0.5s)
		if current_time - last_update >= 0.5 or formatted_status != last_status:
			last_update = current_time
			last_status = formatted_status
			yield formatted_status, plot_update

	# Always yield the final status
	if last_status:
		yield last_status, pd.DataFrame(loss_history) if loss_history else None


def wrapped_stop_training():
	"""Wrapper to format stop training output"""
	from watserface.training import core as training_core
	return training_core.stop_training()


def pre_check() -> bool:
	"""Pre-check for modeler tab"""
	return True


def render() -> gradio.Blocks:
	"""Render the Modeler tab layout"""
	global START_MODELER_BUTTON, STOP_MODELER_BUTTON, MODELER_STATUS, MODELER_LOSS_PLOT

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

		# Load Existing LoRA Option
		lora_loader.render()

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

		# Status Display with Loss Graph
		with gradio.Row():
			MODELER_STATUS = gradio.Textbox(
				label="Training Status",
				value="Idle - Configure settings and click 'Start LoRA Training'",
				interactive=False,
				lines=8,
				elem_id="modeler_training_status",
				scale=1
			)

			MODELER_LOSS_PLOT = gradio.LinePlot(
				label="Training Loss",
				x="epoch", y="loss",
				title="Loss over Time",
				width=400, height=300,
				tooltip=["epoch", "loss"],
				overlay_point=True,
				elem_classes=["loss-chart-container"],
				scale=1
			)

		# Terminal for debugging
		with gradio.Row():
			terminal.render()

		footer.render()

	return layout


def listen() -> None:
	"""Set up event listeners"""
	lora_loader.listen()
	modeler_source.listen()
	modeler_target.listen()
	modeler_options.listen()
	terminal.listen()

	# Wire up training buttons
	START_MODELER_BUTTON.click(
		wrapped_start_lora_training,
		inputs=[
			lora_loader.LORA_MODE_RADIO,
			lora_loader.EXISTING_LORA_DROPDOWN,
			modeler_source.MODELER_SOURCE_PROFILE_DROPDOWN,
			modeler_target.MODELER_TARGET_FILE,
			modeler_options.MODELER_MODEL_NAME,
			modeler_options.MODELER_EPOCHS,
			modeler_options.MODELER_LEARNING_RATE,
			modeler_options.MODELER_LORA_RANK,
			modeler_options.MODELER_BATCH_SIZE
		],
		outputs=[MODELER_STATUS, MODELER_LOSS_PLOT]
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
