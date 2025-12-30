"""
Training Core Module.

Orchestrates dataset extraction, landmark smoothing, and model training.
"""

import os
import shutil
import gradio
from typing import Any

from facefusion import logger, state_manager
from facefusion.filesystem import resolve_relative_path
from facefusion.training.dataset_extractor import extract_training_dataset
from facefusion.training.landmark_smoother import apply_smoothing_to_dataset
from facefusion.training.train_instantid import train_instantid_model
from facefusion.training.train_xseg import train_xseg_model
from facefusion.training.datasets.xseg_dataset import check_dataset_masks


# Global flag for stopping training
_training_stopped = False


def start_identity_training(
	model_name: str,
	epochs: int,
	source_files: Any,
	progress=gradio.Progress()
):
	"""
	Train an identity model (InstantID) from source files.
	"""
	global _training_stopped
	_training_stopped = False
	telemetry = {'status': 'initializing'}

	try:
		if not model_name:
			yield "❌ Error: Please enter a model name.", telemetry
			return
		if not source_files:
			yield "❌ Error: No source files uploaded.", telemetry
			return

		logger.info(f"Starting Identity Training for '{model_name}'...", __name__)

		dataset_path = os.path.join(state_manager.get_item('jobs_path'), 'training_dataset_identity')
		if os.path.exists(dataset_path):
			shutil.rmtree(dataset_path)
		os.makedirs(dataset_path, exist_ok=True)

		source_list = source_files if isinstance(source_files, list) else [source_files]
		source_paths = []
		for source_file in source_list:
			if hasattr(source_file, 'name'):
				source_paths.append(source_file.name)
			else:
				source_paths.append(source_file)

		logger.info(f"Extracting dataset from {len(source_paths)} source file(s)...", __name__)

		# Step 1: Extraction
		last_stats = {}
		for stats in extract_training_dataset(
			source_paths=source_paths,
			output_dir=dataset_path,
			frame_interval=2,
			max_frames=1000,
			progress=progress
		):
			if _training_stopped:
				yield "Training Stopped.", telemetry
				return
			telemetry.update(stats)
			telemetry['status'] = 'Extracting'
			last_stats = stats
			yield f"Extracting... {stats.get('frames_extracted', 0)} frames", telemetry

		if last_stats.get('frames_extracted', 0) == 0:
			yield "❌ Error: No faces found.", telemetry
			return

		# Step 2: Smoothing
		telemetry['status'] = 'Smoothing'
		yield "Applying smoothing...", telemetry
		apply_smoothing_to_dataset(dataset_path)

		# Step 3: Train InstantID model
		telemetry['status'] = 'Training'
		onnx_path = ""
		
		for status_msg, train_stats in train_instantid_model(
			dataset_dir=dataset_path,
			model_name=model_name,
			epochs=epochs,
			batch_size=4,
			learning_rate=0.0001,
			save_interval=max(10, epochs // 5),
			progress=progress
		):
			if _training_stopped:
				yield "Training Stopped.", telemetry
				return
			
			if 'model_path' in train_stats:
				onnx_path = train_stats['model_path']
			else:
				telemetry.update(train_stats)
				yield status_msg, telemetry
	
		if not onnx_path:
			yield "❌ Training failed to produce model.", telemetry
			return
	
		# Copy to assets folder for persistence and detection
		trained_models_dir = resolve_relative_path('../.assets/models/trained')
		if not os.path.exists(trained_models_dir):
			os.makedirs(trained_models_dir, exist_ok=True)
		
		final_model_path = os.path.join(trained_models_dir, os.path.basename(onnx_path))
		shutil.copy(onnx_path, final_model_path)

		# Copy checkpoint for resuming
		checkpoint_src = os.path.splitext(onnx_path)[0] + '.pth'
		checkpoint_dst = os.path.splitext(final_model_path)[0] + '.pth'
		if os.path.exists(checkpoint_src):
			shutil.copy(checkpoint_src, checkpoint_dst)

		# Generate hash file to prevent deletion by validator
		import zlib
		with open(final_model_path, 'rb') as f:
			model_content = f.read()
		model_hash = format(zlib.crc32(model_content), '08x')
		
		final_hash_path = os.path.splitext(final_model_path)[0] + '.hash'
		with open(final_hash_path, 'w') as f:
			f.write(model_hash)

		logger.info(f"Identity model trained successfully and saved to: {final_model_path}", __name__)
		telemetry['status'] = 'Complete'
		telemetry['model_path'] = final_model_path
		yield f"✅ Identity Model '{model_name}' trained successfully!\n📁 Saved to: {final_model_path}\n\n🔄 Refresh the app or click the new refresh button in Swap tab to use it.", telemetry

	except Exception as e:
		logger.error(f"Identity training failed: {e}", __name__)
		import traceback
		traceback.print_exc()
		yield f"❌ Training failed: {str(e)}", telemetry


def start_occlusion_training(
	model_name: str,
	epochs: int,
	target_file: Any,
	progress=gradio.Progress()
):
	"""
	Train an occlusion model (XSeg) from target file.
	"""
	global _training_stopped
	_training_stopped = False
	telemetry = {'status': 'initializing'}

	try:
		if not model_name:
			yield "❌ Error: Please enter a model name.", telemetry
			return
		if not target_file:
			yield "❌ Error: No target file uploaded.", telemetry
			return

		logger.info(f"Starting Occlusion Training for '{model_name}'...", __name__)

		dataset_path = os.path.join(state_manager.get_item('jobs_path'), 'training_dataset_occlusion')
		if os.path.exists(dataset_path):
			shutil.rmtree(dataset_path)
		os.makedirs(dataset_path, exist_ok=True)

		target_path = target_file.name if hasattr(target_file, 'name') else target_file

		logger.info("Extracting dataset from target file...", __name__)

		# Step 1: Extraction
		last_stats = {}
		for stats in extract_training_dataset(
			source_paths=[target_path],
			output_dir=dataset_path,
			frame_interval=2,
			max_frames=2000,
			progress=progress
		):
			if _training_stopped:
				yield "Training Stopped.", telemetry
				return
			telemetry.update(stats)
			telemetry['status'] = 'Extracting'
			last_stats = stats
			yield f"Extracting... {stats.get('frames_extracted', 0)} frames", telemetry

		if last_stats.get('frames_extracted', 0) == 0:
			yield "❌ Error: No frames found.", telemetry
			return

		state_manager.set_item('training_dataset_path', dataset_path)

		# Step 2: Smoothing
		telemetry['status'] = 'Smoothing'
		yield "Applying smoothing...", telemetry
		apply_smoothing_to_dataset(dataset_path)

		# Step 3: Masks
		mask_info = check_dataset_masks(dataset_path)
		telemetry.update(mask_info)
		yield f"Mask Coverage: {mask_info['mask_coverage']:.1f}%", telemetry

		# Step 4: Training
		telemetry['status'] = 'Training'
		onnx_path = ""
		
		for status_msg, train_stats in train_xseg_model(
			dataset_dir=dataset_path,
			model_name=model_name,
			epochs=epochs,
			batch_size=4,
			learning_rate=0.001,
			save_interval=max(10, epochs // 5),
			progress=progress
		):
			if _training_stopped:
				yield "Training Stopped.", telemetry
				return

			if 'model_path' in train_stats:
				onnx_path = train_stats['model_path']
			else:
				telemetry.update(train_stats)
				yield status_msg, telemetry

		if not onnx_path:
			yield "❌ Training failed to produce model.", telemetry
			return

		# Copy...
		trained_models_dir = resolve_relative_path('../.assets/models/trained')
		if not os.path.exists(trained_models_dir):
			os.makedirs(trained_models_dir, exist_ok=True)
		
		final_model_path = os.path.join(trained_models_dir, os.path.basename(onnx_path))
		shutil.copy(onnx_path, final_model_path)
		
		# Hash gen...
		import zlib
		with open(final_model_path, 'rb') as f:
			model_content = f.read()
		model_hash = format(zlib.crc32(model_content), '08x')
		with open(os.path.splitext(final_model_path)[0] + '.hash', 'w') as f:
			f.write(model_hash)

		telemetry['status'] = 'Complete'
		telemetry['model_path'] = final_model_path
		yield "✅ Occlusion Training Complete.", telemetry

	except Exception as e:
		logger.error(f"Occlusion training failed: {e}", __name__)
		import traceback
		traceback.print_exc()
		yield f"❌ Training failed: {str(e)}", telemetry


def stop_training() -> str:
	"""Stop ongoing training (placeholder for future implementation)."""
	global _training_stopped
	if _training_stopped:
		return "Training is already stopped."
	_training_stopped = True
	import inspect
	caller = inspect.stack()[1].function
	logger.info(f"Training stop requested by {caller}", __name__)
	return "⚠️ Training stop requested. Note: Current epoch will complete before stopping."