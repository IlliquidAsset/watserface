"""
Training Core Module.

Orchestrates dataset extraction, landmark smoothing, and model training.
Supports both legacy upload mode and new Face Set mode for reusable datasets.
"""

import os
import shutil
import gradio
import numpy as np
from typing import Any, List, Optional
from pathlib import Path

from watserface import identity_profile, logger, state_manager
from watserface.face_analyser import get_one_face, get_many_faces
from watserface.face_set import get_face_set_manager, FaceSetConfig
from watserface.filesystem import is_video, resolve_file_paths, resolve_relative_path
from watserface.training.dataset_extractor import extract_training_dataset
from watserface.training.landmark_smoother import apply_smoothing_to_dataset
from watserface.training.trainers.identity import train_identity_model
from watserface.training.trainers.xseg import train_xseg_model
from watserface.training.datasets.xseg_dataset import check_dataset_masks
from watserface.vision import read_static_image


# Global flag for stopping training
_training_stopped = False


def start_identity_training(
	model_name: str,
	epochs: int,
	source_files: Any = None,
	face_set_id: Optional[str] = None,
	save_as_face_set: bool = False,
	new_face_set_name: Optional[str] = None,
	progress=gradio.Progress()
):
	"""
	Train an identity model (InstantID) from either Face Set or source files.

	Args:
		model_name: Name for the trained identity model
		epochs: Number of training epochs
		source_files: Source files for upload mode (optional)
		face_set_id: Face Set ID for Face Set mode (optional)
		save_as_face_set: Whether to save extracted data as Face Set
		new_face_set_name: Name for new Face Set (auto-generated if not provided)
		progress: Gradio progress callback

	Yields:
		(status_message, telemetry_dict) tuples
	"""
	global _training_stopped
	_training_stopped = False
	telemetry = {'status': 'initializing'}
	source_paths = []  # Track source paths for profile metadata

	try:
		if not model_name:
			yield ["❌ Error: Please enter a model name.", telemetry]
			return

		logger.info(f"Starting Identity Training for '{model_name}'...", __name__)

		# Determine dataset path based on mode
		dataset_path = None
		face_set_manager = get_face_set_manager()

		if face_set_id:
			# MODE 1: Use existing Face Set
			logger.info(f"Using Face Set: {face_set_id}", __name__)

			face_set = face_set_manager.load_face_set(face_set_id)

			if not face_set:
				yield [f"❌ Error: Face Set '{face_set_id}' not found.", telemetry]
				return

			# Point to Face Set's frames directory
			dataset_path = face_set_manager.get_face_set_frames_path(face_set_id)

			# Track source files from Face Set metadata
			source_paths = face_set.source_files if hasattr(face_set, 'source_files') else []

			telemetry['status'] = 'Using Face Set'
			telemetry['face_set_id'] = face_set_id
			telemetry['frames_count'] = face_set.frame_count
			yield [f"✅ Using Face Set: {face_set.name} ({face_set.frame_count} frames)", telemetry]

			# Skip extraction and smoothing - Face Set already has this done
			logger.info("Skipping extraction - using Face Set frames", __name__)

		elif source_files:
			# MODE 2: Upload new files
			logger.info(f"Upload mode: processing {len(source_files) if isinstance(source_files, list) else 1} file(s)", __name__)

			source_list = source_files if isinstance(source_files, list) else [source_files]
			source_paths = []
			for source_file in source_list:
				if hasattr(source_file, 'name'):
					source_paths.append(source_file.name)
				else:
					source_paths.append(source_file)

			if save_as_face_set:
				# Create Face Set (handles extraction + smoothing)
				face_set_name = new_face_set_name or f"Training_{model_name}_{int(__import__('time').time())}"

				telemetry['status'] = 'Creating Face Set'
				yield [f"Creating Face Set '{face_set_name}'...", telemetry]

				try:
					face_set = face_set_manager.create_face_set(
						source_paths=source_paths,
						name=face_set_name,
						description=f"Created during training of {model_name}",
						tags=["training", model_name],
						progress=progress
					)

					dataset_path = face_set_manager.get_face_set_frames_path(face_set.id)
					telemetry['face_set_created'] = True
					telemetry['face_set_id'] = face_set.id
					telemetry['frames_count'] = face_set.frame_count
					yield [f"✅ Face Set created: {face_set.name} ({face_set.frame_count} frames)", telemetry]

				except Exception as e:
					logger.error(f"Failed to create Face Set: {e}", __name__)
					yield [f"❌ Face Set creation failed: {e}", telemetry]
					return

			else:
				# Traditional temp directory extraction (backward compatible)
				dataset_path = os.path.join(state_manager.get_item('jobs_path'), 'training_dataset_identity')
				os.makedirs(dataset_path, exist_ok=True)

				logger.info(f"Extracting dataset from {len(source_paths)} source file(s)...", __name__)

				# Step 1: Extraction
				last_stats = {}
				existing_frames = len([f for f in os.listdir(dataset_path) if f.endswith('.png')])

				if existing_frames > 0:
					telemetry['status'] = 'Skipping Extraction'
					telemetry['frames_extracted'] = existing_frames
					yield [f"Using {existing_frames} existing frames. Skipping extraction...", telemetry]
					last_stats['frames_extracted'] = existing_frames
				else:
					for stats in extract_training_dataset(
						source_paths=source_paths,
						output_dir=dataset_path,
						frame_interval=2,
						max_frames=1000,
						progress=progress
					):
						if _training_stopped:
							yield ["Training Stopped.", telemetry]
							return
						telemetry.update(stats)
						telemetry['status'] = 'Extracting'
						last_stats = stats
						yield [f"Extracting... {stats.get('frames_extracted', 0)} frames", telemetry]

				if last_stats.get('frames_extracted', 0) == 0:
					yield ["❌ Error: No faces found.", telemetry]
					return

				# Step 2: Smoothing
				telemetry['status'] = 'Smoothing'
				yield ["Applying smoothing...", telemetry]
				apply_smoothing_to_dataset(dataset_path)

		else:
			yield ["❌ Error: Must provide either face_set_id or source_files.", telemetry]
			return

		# Verify frames exist
		if not os.path.exists(dataset_path):
			yield [f"❌ Error: Dataset path does not exist: {dataset_path}", telemetry]
			return

		existing_frames = len([f for f in os.listdir(dataset_path) if f.endswith('.png')])
		if existing_frames == 0:
			yield ["❌ Error: No frames found in dataset.", telemetry]
			return

		logger.info(f"Dataset ready: {existing_frames} frames in {dataset_path}", __name__)

		# Step 3: Train InstantID model
		telemetry['status'] = 'Training'
		onnx_path = ""
		
		# Use a consistent save directory for models to allow resumption
		model_save_dir = os.path.join('models', 'identities', model_name.lower().replace(' ', '_'))
		os.makedirs(model_save_dir, exist_ok=True)

		training_was_stopped = False
		try:
			# Use the new modular trainer (with internal fallback)
			for status_msg, train_stats in train_identity_model(
				dataset_dir=dataset_path,
				model_name=model_name,
				epochs=epochs,
				batch_size=4,
				learning_rate=0.0001,
				save_interval=max(10, epochs // 5),
				save_dir=model_save_dir  # Pass the persistent save directory
			):
				if _training_stopped:
					training_was_stopped = True
					telemetry['status'] = 'Stopped'
					yield ["⚠️ Training stopped by user.", telemetry]
					break  # Break instead of return - allows profile creation to run

				if 'model_path' in train_stats:
					onnx_path = train_stats['model_path']
				else:
					telemetry.update(train_stats)
					yield [status_msg, telemetry]
		except Exception as e:
			error_msg = f"❌ Training Error: {str(e)}"
			logger.error(f"Identity training failed: {e}", __name__)
			import traceback
			traceback.print_exc()
			telemetry['status'] = 'Failed'
			telemetry['error'] = str(e)
			yield [error_msg, telemetry]
			# Don't return - allow profile creation to attempt

		# Copy ONNX model to assets (if training completed)
		final_model_path = None
		if onnx_path:
			# Training completed - copy model
			trained_models_dir = resolve_relative_path('../.assets/models/trained')
			if not os.path.exists(trained_models_dir):
				os.makedirs(trained_models_dir, exist_ok=True)

			final_model_path = os.path.join(trained_models_dir, os.path.basename(onnx_path))
			shutil.copy(onnx_path, final_model_path)

			# Copy checkpoint
			checkpoint_src = os.path.splitext(onnx_path)[0] + '.pth'
			checkpoint_dst = os.path.splitext(final_model_path)[0] + '.pth'
			if os.path.exists(checkpoint_src):
				shutil.copy(checkpoint_src, checkpoint_dst)

			# Generate hash file
			import zlib
			with open(final_model_path, 'rb') as f:
				model_content = f.read()
			model_hash = format(zlib.crc32(model_content), '08x')
			final_hash_path = os.path.splitext(final_model_path)[0] + '.hash'
			with open(final_hash_path, 'w') as f:
				f.write(model_hash)

			logger.info(f"Identity model trained successfully and saved to: {final_model_path}", __name__)
		else:
			logger.info("Training session incomplete - ONNX model not exported yet", __name__)

		# Step 4: Create and save identity profile
		try:
			telemetry['status'] = 'Creating Profile'
			yield ["Creating identity profile...", telemetry]

			# Extract embeddings from all frames in dataset
			frame_paths = resolve_file_paths(dataset_path)
			embeddings = []
			frames_processed = 0

			logger.info(f"Extracting embeddings from {len(frame_paths)} frames (sampling 100)...", __name__)

			for frame_path in frame_paths[:100]:  # Limit to 100 frames for performance
				if frame_path.endswith(('.jpg', '.png')):
					frames_processed += 1
					try:
						frame = read_static_image(frame_path)
						# First detect faces in the frame
						faces = get_many_faces([frame])
						# Then get the first face from the detected faces
						if faces:
							face = get_one_face(faces)
							if face and hasattr(face, 'embedding') and face.embedding is not None:
								embeddings.append(face.embedding)
								logger.debug(f"Extracted embedding from {frame_path}", __name__)
							else:
								logger.debug(f"Face detected but no embedding in {frame_path}", __name__)
						else:
							logger.debug(f"No face detected in {frame_path}", __name__)
					except Exception as e:
						logger.debug(f"Error processing frame {frame_path}: {e}", __name__)
						import traceback
						logger.debug(traceback.format_exc(), __name__)
						continue

			logger.info(f"Processed {frames_processed} frames, extracted {len(embeddings)} embeddings", __name__)

			if embeddings:
				manager = identity_profile.get_identity_manager()
				profile_id = model_name.lower().replace(' ', '_')

				# Try to enrich existing profile first
				new_stats = {
					'total_processed': len(frame_paths),
					'source_count': len(source_paths) if source_paths else 0
				}

				enriched_profile = manager.source_intelligence.enrich_profile(profile_id, embeddings, new_stats)

				if enriched_profile:
					# Enriched existing profile
					profile = enriched_profile
					action = "enriched"
					logger.info(f"✅ Enriched existing identity profile '{model_name}' (now {enriched_profile.quality_stats.get('final_embedding_count')} total embeddings)", __name__)
				else:
					# Create new profile
					embedding_mean = np.mean(embeddings, axis=0).tolist()
					embedding_std = np.std(embeddings, axis=0).tolist()

					profile = identity_profile.IdentityProfile(
						id=profile_id,
						name=model_name,
						created_at=__import__('datetime').datetime.now().isoformat(),
						source_files=source_paths if source_paths else [],
						embedding_mean=embedding_mean,
						embedding_std=embedding_std,
						quality_stats={
							'total_processed': len(frame_paths),
							'final_embedding_count': len(embeddings),
							'source_count': len(source_paths) if source_paths else 0,
							'training_sessions': 1,
							'last_training': __import__('datetime').datetime.now().isoformat()
						},
						is_ephemeral=False
					)
					action = "created"
					logger.info(f"✅ Created new identity profile '{model_name}' with {len(embeddings)} embeddings", __name__)

				# Save profile (create or update)
				manager.source_intelligence.save_profile(profile)

				telemetry['profile_saved'] = True
				telemetry['profile_action'] = action
				telemetry['embedding_count'] = len(embeddings)
				telemetry['total_embeddings'] = profile.quality_stats.get('final_embedding_count', len(embeddings))
				telemetry['training_sessions'] = profile.quality_stats.get('training_sessions', 1)
			else:
				error_msg = f"❌ No embeddings extracted from {frames_processed} frames - profile not created. Check that frames contain visible faces."
				logger.warn(error_msg, __name__)
				telemetry['profile_saved'] = False
				telemetry['error'] = error_msg
		except Exception as e:
			error_msg = f"Failed to create identity profile: {str(e)}"
			logger.error(error_msg, __name__)
			import traceback
			logger.error(traceback.format_exc(), __name__)
			telemetry['profile_saved'] = False
			telemetry['error'] = error_msg

		# Record Face Set usage if training from Face Set
		if face_set_id:
			face_set_manager.record_training_use(face_set_id, model_name, epochs)
			logger.info(f"Recorded training use for Face Set {face_set_id}", __name__)

		# Final status message
		if training_was_stopped:
			telemetry['status'] = 'Stopped'
		elif onnx_path:
			telemetry['status'] = 'Complete'
		else:
			telemetry['status'] = 'Incomplete'

		if final_model_path:
			telemetry['model_path'] = final_model_path

		if telemetry.get('profile_saved'):
			if telemetry.get('profile_action') == 'enriched':
				if onnx_path:
					final_message = f"✅ Identity Model '{model_name}' trained successfully! Profile enriched: +{telemetry['embedding_count']} new embeddings (total: {telemetry['total_embeddings']}, session #{telemetry['training_sessions']})"
				else:
					final_message = f"⚠️ Training stopped early. Profile enriched: +{telemetry['embedding_count']} new embeddings (total: {telemetry['total_embeddings']}, session #{telemetry['training_sessions']}). Now available in Modeler!"
			else:
				if onnx_path:
					final_message = f"✅ Identity Model '{model_name}' trained successfully! Profile created with {telemetry['embedding_count']} embeddings."
				else:
					final_message = f"⚠️ Training stopped early. Profile created with {telemetry['embedding_count']} embeddings. Now available in Modeler!"
		else:
			if onnx_path:
				final_message = f"⚠️ Identity Model '{model_name}' trained, but profile creation failed. {telemetry.get('error', 'Unknown error')}"
			else:
				final_message = f"⚠️ Training stopped early. Profile creation failed: {telemetry.get('error', 'Unknown error')}"

		yield [final_message, telemetry]

	except Exception as e:
		logger.error(f"Identity training failed: {e}", __name__)
		import traceback
		traceback.print_exc()
		yield [f"❌ Training failed: {str(e)}", telemetry]


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
			yield ["❌ Error: Please enter a model name.", telemetry]
			return
		if not target_file:
			yield ["❌ Error: No target file uploaded.", telemetry]
			return

		logger.info(f"Starting Occlusion Training for '{model_name}'...", __name__)

		dataset_path = os.path.join(state_manager.get_item('jobs_path'), 'training_dataset_occlusion')
		os.makedirs(dataset_path, exist_ok=True)

		target_path = target_file.name if hasattr(target_file, 'name') else target_file

		# Step 1: Extraction
		last_stats = {}
		existing_frames = len([f for f in os.listdir(dataset_path) if f.endswith('.png')])
		
		if existing_frames > 0:
			telemetry['status'] = 'Skipping Extraction'
			telemetry['frames_extracted'] = existing_frames
			yield [f"Using {existing_frames} existing frames. Skipping extraction...", telemetry]
			last_stats['frames_extracted'] = existing_frames
		else:
			for stats in extract_training_dataset(
				source_paths=[target_path],
				output_dir=dataset_path,
				frame_interval=2,
				max_frames=2000,
				progress=progress
			):
				if _training_stopped:
					yield ["Training Stopped.", telemetry]
					return
				telemetry.update(stats)
				telemetry['status'] = 'Extracting'
				last_stats = stats
				yield [f"Extracting... {stats.get('frames_extracted', 0)} frames", telemetry]

		if last_stats.get('frames_extracted', 0) == 0:
			yield ["❌ Error: No frames found.", telemetry]
			return

		state_manager.set_item('training_dataset_path', dataset_path)

		# Step 2: Smoothing
		telemetry['status'] = 'Smoothing'
		yield ["Applying smoothing...", telemetry]
		apply_smoothing_to_dataset(dataset_path)

		# Step 3: Masks
		mask_info = check_dataset_masks(dataset_path)
		telemetry.update(mask_info)
		yield [f"Mask Coverage: {mask_info['mask_coverage']:.1f}%", telemetry]

		# Step 4: Training
		telemetry['status'] = 'Training'
		
		# Use a consistent save directory for models to allow resumption
		model_save_dir = os.path.join('models', 'identities', model_name.lower().replace(' ', '_'), 'xseg')
		os.makedirs(model_save_dir, exist_ok=True)
		
		# train_xseg_model is a generator - consume it and extract the final model path
		onnx_path = None
		for status_msg, train_telemetry in train_xseg_model(
			dataset_dir=dataset_path,
			model_name=model_name,
			epochs=epochs,
			batch_size=4,
			learning_rate=0.001,
			save_interval=max(10, epochs // 5),
			progress=progress,
			save_dir=model_save_dir
		):
			if _training_stopped:
				yield ["Training Stopped.", telemetry]
				return
			telemetry.update(train_telemetry)
			yield [status_msg, telemetry]
			# Extract the final model path from telemetry when available
			if 'model_path' in train_telemetry:
				onnx_path = train_telemetry['model_path']
		
		if not onnx_path or not os.path.exists(onnx_path):
			yield ["❌ Error: ONNX export failed - no model path returned.", telemetry]
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
		final_hash_path = os.path.splitext(final_model_path)[0] + '.hash'
		with open(final_hash_path, 'w') as f:
			f.write(model_hash)

		telemetry['status'] = 'Complete'
		telemetry['model_path'] = final_model_path
		yield [f"✅ Occlusion Training Complete.", telemetry]

	except Exception as e:
		logger.error(f"Occlusion training failed: {e}", __name__)
		import traceback
		traceback.print_exc()
		yield [f"❌ Training failed: {str(e)}", telemetry]


def stop_training() -> str:
	"""Stop ongoing training."""
	global _training_stopped
	if _training_stopped:
		return "Training is already stopped."
	_training_stopped = True
	import inspect
	caller = inspect.stack()[1].function
	logger.info(f"Training stop requested by {caller}", __name__)
	return "⚠️ Training stop requested. Note: Current epoch will complete before stopping."
