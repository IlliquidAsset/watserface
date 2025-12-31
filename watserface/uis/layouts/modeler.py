"""
Modeler Tab Layout - Paired Identity Training
"""
from typing import Optional, Any
import gradio
import torch

from watserface import state_manager, logger
from watserface.uis.components import about, footer, modeler_source, modeler_target, modeler_options, terminal, lora_loader


# Training Controls
START_MODELER_BUTTON: Optional[gradio.Button] = None
STOP_MODELER_BUTTON: Optional[gradio.Button] = None
MODELER_STATUS: Optional[gradio.Textbox] = None
MODELER_LOSS_PLOT: Optional[gradio.LinePlot] = None
MODELER_OVERALL_PROGRESS: Optional[gradio.Textbox] = None
MODELER_EPOCH_PROGRESS: Optional[gradio.Textbox] = None


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

		# Check if rank changed - prevents incompatible checkpoint loading
		checkpoint_path = os.path.join('.jobs/training_dataset_lora', f"{existing_lora_name}_lora.pth")
		if not os.path.exists(checkpoint_path):
			checkpoint_path = os.path.join('.assets/models/trained', f"{existing_lora_name}_lora.pth")

		if os.path.exists(checkpoint_path):
			try:
				checkpoint = torch.load(checkpoint_path, map_location='cpu')
				if isinstance(checkpoint, dict):
					saved_rank = checkpoint.get('lora_rank', None)
					if saved_rank and saved_rank != lora_rank:
						yield f"❌ Error: Rank mismatch! Checkpoint has rank {saved_rank}, you selected rank {lora_rank}.\n\nChanging LoRA rank creates a new architecture - you cannot continue training.\n\nOptions:\n1. Keep rank {saved_rank} to continue training {existing_lora_name}\n2. Use rank {lora_rank} with a NEW model name (e.g., '{existing_lora_name}_rank{lora_rank}')", None
						return
			except:
				pass  # If can't load checkpoint, proceed anyway

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
		yield f"📂 Using {existing_frames} existing frames. Skipping extraction...", None, "Preparing...", "Preparing..."
	else:
		# Step 1: Extract frames from target
		yield "📹 Extracting frames from target video...", None, "Extracting...", "Extracting..."
		for stats in extract_training_dataset(
			source_paths=[target_path],
			output_dir=dataset_path,
			frame_interval=2,
			max_frames=1000
		):
			yield f"Extracting... {stats.get('frames_extracted', 0)} frames", None, "Extracting...", "Extracting..."

		# Step 2: Apply smoothing
		yield "🎨 Applying landmark smoothing...", None, "Smoothing...", "Smoothing..."
		apply_smoothing_to_dataset(dataset_path)

	# Step 3: Train LoRA model
	yield "🚀 Starting LoRA training...", None, "Starting...", "Starting..."

	last_update = 0
	last_status = None
	last_plot = None

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
			frames_used = telemetry.get('frames_used', 0)
			batch_size_actual = telemetry.get('batch_size', batch_size)

			# Calculate progress
			try:
				epoch_num = int(epoch)
				total_epochs_num = int(total_epochs)
				overall_pct = (epoch_num / total_epochs_num * 100) if total_epochs_num > 0 else 0
				overall_progress_str = f"{epoch_num}/{total_epochs_num} epochs ({overall_pct:.1f}%)"
			except:
				overall_progress_str = f"{epoch}/{total_epochs} epochs"

			# Epoch progress (batches)
			batches_per_epoch = (frames_used // batch_size_actual) if batch_size_actual > 0 else 0
			epoch_progress_str = f"Batch progress tracked per epoch ({batches_per_epoch} batches/epoch)"

			formatted_status = f"""📊 LoRA Training Progress

Epoch: {epoch}/{total_epochs}
Loss: {loss}
ETA: {eta}
Device: {device}
LoRA Rank: {rank}
Trainable Parameters: {trainable:,}

{message}"""

			# Track loss for graphing (use history from telemetry)
			if telemetry.get('loss_history'):
				try:
					plot_update = pd.DataFrame(telemetry['loss_history'])
				except:
					plot_update = None
			else:
				plot_update = None
		else:
			formatted_status = message
			plot_update = None
			overall_progress_str = "Initializing..."
			epoch_progress_str = "Waiting..."

		# Throttle UI updates to avoid flickering (every 0.5s)
		if current_time - last_update >= 0.5 or formatted_status != last_status:
			last_update = current_time
			last_status = formatted_status
			last_plot = plot_update
			yield formatted_status, plot_update, overall_progress_str, epoch_progress_str

	# Always yield the final status
	if last_status:
		yield last_status, last_plot, "Complete!", "Complete!"


def wrapped_stop_training():
	"""Wrapper to format stop training output"""
	from watserface.training import core as training_core
	stop_msg = training_core.stop_training()

	# Provide detailed feedback about what happens next
	return f"""⚠️ Training Stop Requested

Current epoch will complete, then:
1. 💾 Saving final checkpoint...
2. 🔄 Exporting ONNX model with merged LoRA weights...
3. 📦 Moving model to .assets/models/trained/
4. ✅ Training complete!

Please wait for export to finish..."""


def update_model_name_visibility(mode: str, existing_lora: str = None) -> gradio.Textbox:
	"""Update model name input based on training mode"""
	if mode == "Continue Training":
		if existing_lora:
			return gradio.Textbox(value=existing_lora, interactive=False, label="Model Name (auto-filled from selected LoRA)")
		else:
			return gradio.Textbox(value="", interactive=False, label="Model Name (select LoRA first)")
	else:
		return gradio.Textbox(value="", interactive=True, label="Model Name")


def update_model_name_from_lora(existing_lora: str = None) -> gradio.Textbox:
	"""Update model name when LoRA is selected"""
	if existing_lora:
		return gradio.Textbox(value=existing_lora, interactive=False, label="Model Name (auto-filled from selected LoRA)")
	else:
		return gradio.Textbox(value="", interactive=False, label="Model Name (select LoRA first)")


def pre_check() -> bool:
	"""Pre-check for modeler tab"""
	return True


def render() -> gradio.Blocks:
	"""Render the Modeler tab layout"""
	global START_MODELER_BUTTON, STOP_MODELER_BUTTON, MODELER_STATUS, MODELER_LOSS_PLOT, MODELER_OVERALL_PROGRESS, MODELER_EPOCH_PROGRESS

	with gradio.Blocks() as layout:
		about.render()

		gradio.Markdown("## 🎯 Step 3: LoRA Paired Training (Modeler)")

		with gradio.Accordion("ℹ️ How it works", open=False):
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

		# Load Existing LoRA Option (at top)
		lora_loader.render()

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

		# Progress Bars
		with gradio.Row():
			with gradio.Column(scale=1):
				MODELER_OVERALL_PROGRESS = gradio.Textbox(
					label="⏱️ Overall Progress",
					value="0/0 epochs (0%)",
					interactive=False,
					lines=1,
					elem_id="modeler_overall_progress"
				)
				MODELER_EPOCH_PROGRESS = gradio.Textbox(
					label="📊 Current Epoch Progress",
					value="0/0 batches",
					interactive=False,
					lines=1,
					elem_id="modeler_epoch_progress"
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

		# Status Display
		MODELER_STATUS = gradio.Textbox(
			label="Training Status",
			value="Idle - Configure settings and click 'Start LoRA Training'",
			interactive=False,
			lines=6,
			elem_id="modeler_training_status"
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

	# Disable model name when continuing training
	lora_loader.LORA_MODE_RADIO.change(
		update_model_name_visibility,
		inputs=[lora_loader.LORA_MODE_RADIO, lora_loader.EXISTING_LORA_DROPDOWN],
		outputs=[modeler_options.MODELER_MODEL_NAME]
	)

	lora_loader.EXISTING_LORA_DROPDOWN.change(
		update_model_name_from_lora,
		inputs=[lora_loader.EXISTING_LORA_DROPDOWN],
		outputs=[modeler_options.MODELER_MODEL_NAME]
	)

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
		outputs=[MODELER_STATUS, MODELER_LOSS_PLOT, MODELER_OVERALL_PROGRESS, MODELER_EPOCH_PROGRESS]
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
