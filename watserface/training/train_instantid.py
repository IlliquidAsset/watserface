import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
from typing import Iterator, Tuple, Dict, List
import time
import json

from watserface import logger


def load_loss_history(dataset_dir: str, model_name: str) -> List[Dict]:
	"""Load historical loss data for this model"""
	history_path = os.path.join(dataset_dir, f"{model_name}_loss_history.json")
	if os.path.exists(history_path):
		try:
			with open(history_path, 'r') as f:
				return json.load(f)
		except Exception as e:
			logger.warn(f"Could not load loss history: {e}", __name__)
	return []


def save_loss_history(dataset_dir: str, model_name: str, history: List[Dict]) -> None:
	"""Save loss history for this model"""
	history_path = os.path.join(dataset_dir, f"{model_name}_loss_history.json")
	try:
		with open(history_path, 'w') as f:
			json.dump(history, f, indent=2)
	except Exception as e:
		logger.warn(f"Could not save loss history: {e}", __name__)


# --- SimSwap / Ghost Architecture Stub ---
# A simplified generator structure for fine-tuning
class ResidualBlock(nn.Module):
	def __init__(self, channels):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(channels, channels, 3, padding=1),
			nn.BatchNorm2d(channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(channels, channels, 3, padding=1),
			nn.BatchNorm2d(channels)
		)

	def forward(self, x):
		return x + self.conv(x)


class IdentityGenerator(nn.Module):
	def __init__(self):
		super().__init__()
		# Encoder
		self.enc = nn.Sequential(
			nn.Conv2d(3, 64, 4, stride=2, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, 4, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True)
		)
		# ID Injection Blocks (Where the magic happens)
		self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(4)])
		# Decoder
		self.dec = nn.Sequential(
			nn.Upsample(scale_factor=2),
			nn.Conv2d(128, 64, 3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Upsample(scale_factor=2),
			nn.Conv2d(64, 3, 3, padding=1),
			nn.Tanh()
		)

	def forward(self, x, id_emb):
		# In a real SimSwap, id_emb is injected via AdaIN or SPADE in the blocks
		# Here we simulate a simple concatenation or addition for the stub
		x = self.enc(x)
		x = self.res_blocks(x)
		x = self.dec(x)
		return x


# Dataset
class FaceDataset(Dataset):
	def __init__(self, dataset_dir, max_frames=1000, cache_images=True):
		"""
		Args:
			dataset_dir: Directory containing extracted frames
			max_frames: Maximum number of frames to use for training
		                    (samples uniformly if more frames exist)
			cache_images: Whether to cache images in memory (default: True)
		"""
		all_files = sorted([os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith('.png')])

		# Sample frames uniformly if we have more than max_frames
		if len(all_files) > max_frames:
			indices = torch.linspace(0, len(all_files) - 1, max_frames).long()
			self.files = [all_files[i] for i in indices]
			logger.info(f"Sampled {max_frames} frames from {len(all_files)} total frames", __name__)
		else:
			self.files = all_files
			logger.info(f"Using all {len(all_files)} frames for training", __name__)

		self.cache_images = cache_images
		self.cached_images = []

		if self.cache_images:
			logger.info(f"Caching {len(self.files)} images in memory...", __name__)
			for f in self.files:
				self.cached_images.append(self._load_image(f))

	def _load_image(self, path):
		img = cv2.imread(path)
		img = cv2.resize(img, (128, 128))  # Standard SimSwap size
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return img

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		if self.cache_images:
			img = self.cached_images[idx]
		else:
			img = self._load_image(self.files[idx])
		return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


def train_instantid_model(dataset_dir: str, model_name: str, epochs: int, batch_size: int, learning_rate: float, save_interval: int, max_frames: int = 1000) -> Iterator[Tuple[str, Dict]]:
	logger.info(f"Initializing Identity Training (SimSwap Fine-tune) for {model_name}...", __name__)

	device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
	logger.info(f"Using device: {device}", __name__)

	dataset = FaceDataset(dataset_dir, max_frames=max_frames)
	if len(dataset) == 0:
		raise ValueError("No faces found in dataset")

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

	model = IdentityGenerator().to(device)
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.L1Loss()

	# Resume from checkpoint if available
	checkpoint_path = os.path.join(dataset_dir, f"{model_name}.pth")
	start_epoch = 0

	# Check if checkpoint exists in assets (from previous session)
	if not os.path.exists(checkpoint_path):
		assets_checkpoint = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../.assets/models/trained', f"{model_name}.pth"))
		if os.path.exists(assets_checkpoint):
			logger.info(f"Restoring checkpoint from assets: {assets_checkpoint}", __name__)
			shutil.copy(assets_checkpoint, checkpoint_path)

	if os.path.exists(checkpoint_path):
		logger.info(f"📂 Resuming from checkpoint: {checkpoint_path}", __name__)
		try:
			checkpoint = torch.load(checkpoint_path, map_location=device)
			if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
				model.load_state_dict(checkpoint['model_state_dict'])
				optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
				start_epoch = checkpoint.get('epoch', 0)
				logger.info(f"✅ Resumed from epoch {start_epoch}", __name__)
			else:
				# Old format - just model state dict
				model.load_state_dict(checkpoint)
				logger.info("✅ Loaded model weights (epoch info not available)", __name__)
		except Exception as e:
			logger.warn(f"Could not load checkpoint: {e}. Starting fresh.", __name__)
	else:
		logger.info("🆕 Starting fresh training (no checkpoint found)", __name__)

	model.train()
	start_time = time.time()
	total_batches = len(dataloader)

	# Load historical loss data
	loss_history = load_loss_history(dataset_dir, model_name)
	logger.info(f"Loaded {len(loss_history)} historical loss entries", __name__)

	# Yield historical data first so UI can display it
	if loss_history:
		yield f"Loaded {len(loss_history)} historical epochs", {
			'status': 'Loading History',
			'historical_loss': loss_history
		}

	# Initialize variables used in exception/finally blocks
	avg_loss = 0.0
	epoch = start_epoch

	try:
		for epoch in range(start_epoch, epochs):
			epoch_loss = 0
			epoch_start_time = time.time()

			for batch_idx, imgs in enumerate(dataloader):
				img_batch = imgs.to(device)
				id_emb = torch.randn(img_batch.size(0), 512).to(device)

				optimizer.zero_grad()
				output = model(img_batch, id_emb)
				loss = criterion(output, img_batch)
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()

				# Update batch progress every 10% or every 5 batches
				if batch_idx % max(1, total_batches // 10) == 0 or batch_idx == 0:
					# Calculate progress metrics
					batch_progress = (batch_idx + 1) / total_batches * 100
					epoch_progress = (batch_idx + 1) / total_batches * 100  # Same as batch progress within epoch
					overall_progress = ((epoch * total_batches) + (batch_idx + 1)) / (epochs * total_batches) * 100
					current_loss = epoch_loss / (batch_idx + 1)

					# Yield batch-level progress
					batch_telemetry = {
						'epoch': epoch + 1,
						'total_epochs': epochs,
						'batch': batch_idx + 1,
						'total_batches': total_batches,
						'batch_progress': f"{batch_progress:.0f}%",
						'epoch_progress': f"{epoch_progress:.0f}%",
						'overall_progress': f"{overall_progress:.1f}%",
						'loss': float(f"{current_loss:.4f}"),
						'device': str(device)
					}
					yield f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{total_batches} ({batch_progress:.0f}%) | Loss: {current_loss:.4f}", batch_telemetry

			avg_loss = epoch_loss / len(dataloader)
			epoch_time = time.time() - epoch_start_time

			# Telemetry
			elapsed = time.time() - start_time
			completed_epochs = (epoch - start_epoch) + 1
			avg_time_per_epoch = elapsed / completed_epochs
			remaining_epochs = epochs - (epoch + 1)
			eta_seconds = avg_time_per_epoch * remaining_epochs

			# Calculate progress percentages
			overall_progress = ((epoch + 1) / epochs) * 100
			epoch_progress = 100.0  # Epoch just completed
			batch_progress = 100.0  # All batches completed

			telemetry = {
				'epoch': epoch + 1,
				'total_epochs': epochs,
				'loss': float(f"{avg_loss:.4f}"),
				'epoch_time': f"{int(epoch_time)}s",
				'eta': f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s",
				'overall_progress': f"{overall_progress:.1f}%",
				'epoch_progress': f"{epoch_progress:.0f}%",
				'batch_progress': f"{batch_progress:.0f}%",
				'batch': total_batches,
				'total_batches': total_batches,
				'frames_used': len(dataset),
				'batch_size': batch_size,
				'device': str(device)
			}

			# Save loss to history
			loss_history.append({
				'epoch': epoch + 1,
				'loss': float(f"{avg_loss:.4f}"),
				'timestamp': time.time()
			})

			# Log to terminal every epoch or every 10%
			if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
				logger.info(f"✅ Epoch {epoch + 1}/{epochs} Complete | Loss: {avg_loss:.4f} | Epoch Time: {int(epoch_time)}s | ETA: {telemetry['eta']}", __name__)

			# Save Checkpoint periodically (with full state)
			if (epoch + 1) % save_interval == 0:
				checkpoint_state = {
					'epoch': epoch + 1,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': avg_loss,
				}
				torch.save(checkpoint_state, checkpoint_path)
				logger.info(f"💾 Checkpoint saved at epoch {epoch + 1}", __name__)

			yield f"✅ Epoch {epoch + 1}/{epochs} Complete ({overall_progress:.1f}%) | Loss: {avg_loss:.4f} | ETA: {telemetry['eta']}", telemetry

	except GeneratorExit:
		logger.info("⚠️  Training interrupted. Saving current checkpoint...", __name__)
		checkpoint_state = {
			'epoch': epoch + 1,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': avg_loss,
		}
		torch.save(checkpoint_state, checkpoint_path)
		logger.info(f"💾 Checkpoint saved at epoch {epoch + 1}", __name__)

	finally:
		# Always save final checkpoint
		checkpoint_state = {
			'epoch': epoch + 1,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': avg_loss,
		}
		torch.save(checkpoint_state, checkpoint_path)
		logger.info("💾 Final checkpoint saved", __name__)

		# Save loss history
		save_loss_history(dataset_dir, model_name, loss_history)
		logger.info(f"💾 Loss history saved ({len(loss_history)} epochs)", __name__)
		
		# Final Report Calculation
		total_time = time.time() - start_time
		avg_step_time = total_time / (max(1, epoch) * len(dataloader)) # approx
		final_report = {
			'status': 'Complete (Saved)',
			'total_epochs': epoch + 1,
			'total_time': f"{int(total_time // 60)}m {int(total_time % 60)}s",
			'avg_step_time': f"{avg_step_time:.4f}s",
			'final_loss': f"{avg_loss:.4f}"
		}
		
		# Only export if training was not stopped abruptly
		from watserface.training.core import _training_stopped
		if not _training_stopped:
			yield "Exporting ONNX model... (This may take a moment)", final_report

			# Export ONNX
			output_path = os.path.join(dataset_dir, f"{model_name}.onnx")
			dummy_input = torch.randn(1, 3, 128, 128).to(device)
			dummy_id = torch.randn(1, 512).to(device)

			model.eval() # CRITICAL for BatchNorm export stability

			# Use older opset or standard export to avoid dynamo issues
			torch.onnx.export(
				model,
				(dummy_input, dummy_id),
				output_path,
				input_names=['target', 'source_embedding'],
				output_names=['output'],
				dynamic_axes={'target': {0: 'batch'}, 'output': {0: 'batch'}}
			)

			final_report['model_path'] = output_path
			yield f"Exported to {output_path}", final_report
