"""
LoRA Training Module - Paired Source→Target Fine-Tuning

Trains lightweight LoRA adapters for specific source→target mappings.
Uses frozen base model + trainable LoRA layers for efficient fine-tuning.
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Any, Iterator, Tuple, Dict, Optional
import time

from watserface import logger
from watserface.training.models.lora_adapter import LoRAWrapper
from watserface.training.datasets.lora_dataset import LoRAPairedDataset, LoRACollator, create_lora_dataset
from watserface.training.train_instantid import IdentityGenerator  # Reuse base model


def train_lora_model(
	dataset_dir: str,
	source_profile_id: str,
	model_name: str,
	epochs: int,
	batch_size: int,
	learning_rate: float,
	lora_rank: int,
	save_interval: int,
	progress: Any = None,
	max_frames: int = 1000
) -> Iterator[Tuple[str, Dict]]:
	"""
	Train a LoRA adapter for source→target face swapping.

	Args:
		dataset_dir: Directory with extracted target frames
		source_profile_id: ID of source identity profile
		model_name: Name for the trained LoRA model
		epochs: Number of training epochs
		batch_size: Training batch size
		learning_rate: Learning rate for optimizer
		lora_rank: LoRA rank (4-128, lower = fewer params)
		save_interval: Save checkpoint every N epochs
		progress: Gradio progress callback
		max_frames: Maximum frames to use from dataset

	Yields:
		(status_message, telemetry_dict)
	"""
	logger.info(f"Initializing LoRA Training for '{model_name}'...", __name__)

	# Normalize model name to ensure consistent _lora suffix
	if not model_name.endswith('_lora'):
		full_model_name = f"{model_name}_lora"
	else:
		full_model_name = model_name
	
	logger.info(f"Full model name: {full_model_name}", __name__)

	# Device detection
	device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
	logger.info(f"Using device: {device}", __name__)

	# Create dataset
	dataset = create_lora_dataset(
		dataset_dir=dataset_dir,
		source_profile_id=source_profile_id,
		max_frames=max_frames,
		frame_size=128
	)

	if dataset is None or len(dataset) == 0:
		raise ValueError("Failed to create LoRA dataset or no frames found")

	# Log dataset stats
	stats = dataset.get_batch_stats(batch_size)
	logger.info(f"[LoRA] Dataset: {stats['total_frames']} frames, {stats['batches_per_epoch']} batches/epoch", __name__)

	# Create dataloader
	collator = LoRACollator()
	dataloader = DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=collator,
		num_workers=0  # MPS doesn't support multi-worker
	)

	# Load base model (frozen)
	base_model = IdentityGenerator().to(device)
	logger.info("[LoRA] Base model loaded (will be frozen)", __name__)

	# Add LoRA layers (auto-detect all Conv2d and Linear layers)
	model = LoRAWrapper.add_lora_to_model(
		base_model,
		target_modules=None,  # Auto-detect Conv2d and Linear layers
		rank=lora_rank,
		alpha=float(lora_rank),  # Typically alpha = rank
		dropout=0.1
	)

	# Count parameters
	trainable, total, pct = LoRAWrapper.count_lora_parameters(model)
	logger.info(f"[LoRA] Trainable: {trainable:,} / {total:,} ({pct:.2f}%)", __name__)

	# Optimizer (only LoRA parameters)
	optimizer = optim.AdamW(
		[p for p in model.parameters() if p.requires_grad],
		lr=learning_rate,
		weight_decay=0.01  # Slight regularization
	)

	# Loss function
	criterion = nn.L1Loss()  # Reconstruction loss

	# Checkpoint management using the normalized name
	checkpoint_path = os.path.join(dataset_dir, f"{full_model_name}.pth")
	start_epoch = 0

	# Check for existing checkpoint in assets
	if not os.path.exists(checkpoint_path):
		assets_checkpoint = os.path.abspath(
			os.path.join(os.path.dirname(__file__), '../../.assets/models/trained', f"{full_model_name}.pth")
		)
		if os.path.exists(assets_checkpoint):
			logger.info(f"📂 Restoring checkpoint from assets: {assets_checkpoint}", __name__)
			shutil.copy(assets_checkpoint, checkpoint_path)

	# Load checkpoint if exists
	loss_history = []
	if os.path.exists(checkpoint_path):
		logger.info(f"📂 Resuming from checkpoint: {checkpoint_path}", __name__)
		try:
			checkpoint = torch.load(checkpoint_path, map_location=device)
			if isinstance(checkpoint, dict):
				LoRAWrapper.load_lora_state_dict(model, checkpoint.get('lora_state', {}))
				if 'optimizer_state_dict' in checkpoint:
					optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
				start_epoch = checkpoint.get('epoch', 0)
				loss_history = checkpoint.get('loss_history', [])
				logger.info(f"✅ Resumed from epoch {start_epoch} with {len(loss_history)} historical loss points", __name__)
			else:
				logger.warn("Checkpoint format not recognized, starting fresh", __name__)
		except Exception as e:
			logger.warn(f"Could not load checkpoint: {e}. Starting fresh.", __name__)
	else:
		logger.info(f"🆕 Starting fresh LoRA training (no checkpoint found)", __name__)

	model.train()
	start_time = time.time()

	# Initialize variables used in exception/finally blocks
	avg_loss = 0.0
	epoch = start_epoch

	try:
		# Handle progress being None (when called from UI wrapper)
		epoch_iterator = range(start_epoch, epochs)
		if progress is not None:
			epoch_iterator = progress.tqdm(epoch_iterator, desc=f"Training LoRA (rank={lora_rank})")

		for epoch in epoch_iterator:
			epoch_loss = 0
			batch_count = 0

			for batch in dataloader:
				source_embs = batch['source_embeddings'].to(device)
				target_frames = batch['target_frames'].to(device)
				# landmarks = batch['target_landmarks'].to(device)  # Future: use for conditioning

				optimizer.zero_grad()

				# Forward pass: try to reconstruct target from source embedding
				# In a real face swapper, this would be: swap(source_emb, target_frame)
				# For now, we use the base model architecture
				output = model(target_frames, source_embs)

				# Reconstruction loss: output should match target
				loss = criterion(output, target_frames)

				loss.backward()
				optimizer.step()

				epoch_loss += loss.item()
				batch_count += 1

			avg_loss = epoch_loss / batch_count

			# Track loss history
			loss_history.append({'epoch': epoch + 1, 'loss': avg_loss})

			# Telemetry
			elapsed = time.time() - start_time
			completed_epochs = (epoch - start_epoch) + 1
			avg_time_per_epoch = elapsed / completed_epochs
			remaining_epochs = epochs - (epoch + 1)
			eta_seconds = avg_time_per_epoch * remaining_epochs

			telemetry = {
				'epoch': epoch + 1,
				'total_epochs': epochs,
				'loss': f"{avg_loss:.4f}",
				'eta': f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s",
				'lora_rank': lora_rank,
				'trainable_params': trainable,
				'frames_used': len(dataset),
				'batch_size': batch_size,
				'device': str(device),
				'loss_history': loss_history  # Include full history in telemetry
			}

			# Log to terminal every 10% or last epoch
			if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
				logger.info(
					f"LoRA Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | ETA: {telemetry['eta']}",
					__name__
				)

			# Save checkpoint periodically
			if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
				checkpoint_state = {
					'epoch': epoch + 1,
					'lora_state': LoRAWrapper.extract_lora_state_dict(model),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': avg_loss,
					'lora_rank': lora_rank,
					'source_profile_id': source_profile_id,
					'loss_history': loss_history  # Save full history in checkpoint
				}
				torch.save(checkpoint_state, checkpoint_path)
				logger.info(f"💾 LoRA checkpoint saved at epoch {epoch + 1}", __name__)

			yield f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | ETA: {telemetry['eta']}", telemetry

	except GeneratorExit:
		logger.info("⚠️  LoRA training interrupted. Saving checkpoint...", __name__)
		checkpoint_state = {
			'epoch': epoch + 1,
			'lora_state': LoRAWrapper.extract_lora_state_dict(model),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': avg_loss,
			'lora_rank': lora_rank,
			'source_profile_id': source_profile_id,
			'loss_history': loss_history
		}
		torch.save(checkpoint_state, checkpoint_path)
		logger.info(f"💾 LoRA checkpoint saved at epoch {epoch + 1}", __name__)

	finally:
		# Always save final checkpoint
		checkpoint_state = {
			'epoch': epoch + 1,
			'lora_state': LoRAWrapper.extract_lora_state_dict(model),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': avg_loss,
			'lora_rank': lora_rank,
			'source_profile_id': source_profile_id,
			'loss_history': loss_history
		}
		torch.save(checkpoint_state, checkpoint_path)
		logger.info(f"💾 Final LoRA checkpoint saved", __name__)

		# Final report
		total_time = time.time() - start_time
		final_report = {
			'status': 'Complete (Saved)',
			'total_epochs': epoch + 1,
			'total_time': f"{int(total_time // 60)}m {int(total_time % 60)}s",
			'final_loss': f"{avg_loss:.4f}",
			'lora_params': f"{trainable:,}",
			'compression_ratio': f"{pct:.2f}%"
		}

		yield "🔄 Exporting ONNX model with merged LoRA weights...", final_report

		# Export to ONNX
		try:
			from watserface.filesystem import resolve_relative_path

			output_path = os.path.join(dataset_dir, f"{full_model_name}.onnx")
			model.eval()

			dummy_target = torch.randn(1, 3, 128, 128).to(device)
			dummy_source = torch.randn(1, 512).to(device)

			# Yield progress during export
			yield "🔄 Creating ONNX graph...", final_report

			torch.onnx.export(
				model,
				(dummy_target, dummy_source),
				output_path,
				input_names=['target_frame', 'source_embedding'],
				output_names=['swapped_face'],
				dynamic_axes={
					'target_frame': {0: 'batch'},
					'swapped_face': {0: 'batch'}
				},
				opset_version=12
			)

			logger.info(f"✅ LoRA model exported to: {output_path}", __name__)

			# Yield progress during copy
			yield "📦 Moving to trained models directory...", final_report

			# Copy to assets/models/trained for use in face swapper
			trained_models_dir = resolve_relative_path('../.assets/models/trained')
			if not os.path.exists(trained_models_dir):
				os.makedirs(trained_models_dir, exist_ok=True)

			final_model_path = os.path.join(trained_models_dir, f"{full_model_name}.onnx")
			shutil.copy(output_path, final_model_path)

			# Copy external data file if it exists (.onnx.data)
			external_data_path = output_path + '.data'
			if os.path.exists(external_data_path):
				final_data_path = final_model_path + '.data'
				shutil.copy(external_data_path, final_data_path)
				logger.info(f"✅ External data file copied: {final_data_path}", __name__)

			# Also copy the checkpoint
			checkpoint_dst = os.path.join(trained_models_dir, f"{full_model_name}.pth")
			if os.path.exists(checkpoint_path):
				shutil.copy(checkpoint_path, checkpoint_dst)

			# Generate hash file
			import zlib
			with open(final_model_path, 'rb') as f:
				model_content = f.read()
			model_hash = format(zlib.crc32(model_content), '08x')
			final_hash_path = os.path.splitext(final_model_path)[0] + '.hash'
			with open(final_hash_path, 'w') as f:
				f.write(model_hash)

			logger.info(f"✅ LoRA model copied to assets: {final_model_path}", __name__)

			final_report['model_path'] = final_model_path
			yield f"✅ Export Complete!\n\nModel saved to:\n{final_model_path}\n\nYou can now use '{model_name}' in the Swap tab.", final_report

		except Exception as e:
			logger.error(f"ONNX export failed: {e}", __name__)
			import traceback
			traceback.print_exc()
			yield f"⚠️  Training complete, but ONNX export failed: {e}", final_report