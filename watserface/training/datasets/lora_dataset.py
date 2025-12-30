"""
LoRA Paired Dataset Loader

Loads paired data for training source→target LoRA models:
- Source: Identity profile embeddings (fixed)
- Target: Video frames with detected faces

Dataset structure:
    training_dataset_lora/
    ├── frames/
    │   ├── 000001.png
    │   ├── 000002.png
    │   └── ...
    ├── landmarks/
    │   ├── 000001.npy  (478 MediaPipe landmarks)
    │   └── ...
    └── source_embedding.npy  (Mean embedding from profile)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple
from pathlib import Path

from watserface import logger


class LoRAPairedDataset(Dataset):
	"""
	Dataset for training LoRA models with paired source→target data.
	"""

	def __init__(
		self,
		dataset_dir: str,
		source_embedding: np.ndarray,
		max_frames: int = 1000,
		frame_size: int = 128,
		augment: bool = True
	):
		"""
		Args:
			dataset_dir: Directory containing extracted frames and landmarks
			source_embedding: Source identity embedding (512-dim)
			max_frames: Maximum frames to use (samples uniformly if more exist)
			frame_size: Target frame size for training (128x128 or 256x256)
			augment: Whether to apply data augmentation
		"""
		self.dataset_dir = dataset_dir
		self.source_embedding = torch.from_numpy(source_embedding).float()
		self.frame_size = frame_size
		self.augment = augment

		# Load frame paths
		frames_dir = os.path.join(dataset_dir, 'frames') if os.path.exists(os.path.join(dataset_dir, 'frames')) else dataset_dir
		all_frames = sorted([
			os.path.join(frames_dir, f)
			for f in os.listdir(frames_dir)
			if f.endswith(('.png', '.jpg'))
		])

		# Sample frames if too many
		if len(all_frames) > max_frames:
			indices = torch.linspace(0, len(all_frames) - 1, max_frames).long()
			self.frame_paths = [all_frames[i] for i in indices]
			logger.info(f"[LoRA Dataset] Sampled {max_frames} frames from {len(all_frames)} total", __name__)
		else:
			self.frame_paths = all_frames
			logger.info(f"[LoRA Dataset] Using all {len(all_frames)} frames", __name__)

		# Load corresponding landmarks if available
		landmarks_dir = os.path.join(dataset_dir, 'landmarks')
		self.has_landmarks = os.path.exists(landmarks_dir)

		if self.has_landmarks:
			self.landmark_paths = [
				os.path.join(landmarks_dir, Path(fp).stem + '.npy')
				for fp in self.frame_paths
			]
		else:
			logger.warn("[LoRA Dataset] No landmarks directory found - training without landmark conditioning", __name__)
			self.landmark_paths = None

		logger.info(f"[LoRA Dataset] Initialized with {len(self.frame_paths)} frame pairs", __name__)

	def __len__(self):
		return len(self.frame_paths)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Returns:
			source_emb: Source identity embedding (512,)
			target_frame: Target face frame (3, H, W)
			target_landmarks: Target face landmarks (478, 3) or zeros if not available
		"""
		# Load source embedding (same for all samples)
		source_emb = self.source_embedding

		# Load target frame
		target_frame = cv2.imread(self.frame_paths[idx])
		target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)
		target_frame = cv2.resize(target_frame, (self.frame_size, self.frame_size))

		# Data augmentation
		if self.augment and np.random.rand() > 0.5:
			# Random horizontal flip
			target_frame = cv2.flip(target_frame, 1)

		# Normalize to [-1, 1]
		target_frame = (target_frame.astype(np.float32) / 127.5) - 1.0
		target_frame = torch.from_numpy(target_frame).permute(2, 0, 1).float()

		# Load landmarks if available
		if self.has_landmarks and self.landmark_paths:
			try:
				landmarks = np.load(self.landmark_paths[idx])
				# Normalize landmarks to [-1, 1] range
				landmarks = (landmarks - landmarks.mean(axis=0)) / (landmarks.std(axis=0) + 1e-6)
				target_landmarks = torch.from_numpy(landmarks).float()
			except Exception as e:
				logger.debug(f"Failed to load landmarks for {self.frame_paths[idx]}: {e}", __name__)
				target_landmarks = torch.zeros(478, 3)
		else:
			target_landmarks = torch.zeros(478, 3)

		return source_emb, target_frame, target_landmarks

	def get_batch_stats(self, batch_size: int) -> dict:
		"""Get dataset statistics for logging"""
		return {
			'total_frames': len(self.frame_paths),
			'batches_per_epoch': len(self.frame_paths) // batch_size,
			'source_embedding_dim': self.source_embedding.shape[0],
			'frame_size': self.frame_size,
			'has_landmarks': self.has_landmarks,
			'augmentation': self.augment
		}


class LoRACollator:
	"""
	Custom collate function for batching LoRA dataset samples.
	Handles variable-length landmarks and ensures proper batching.
	"""

	def __call__(self, batch):
		"""
		Args:
			batch: List of (source_emb, target_frame, target_landmarks) tuples

		Returns:
			Batched tensors
		"""
		source_embs = torch.stack([item[0] for item in batch])
		target_frames = torch.stack([item[1] for item in batch])
		target_landmarks = torch.stack([item[2] for item in batch])

		return {
			'source_embeddings': source_embs,
			'target_frames': target_frames,
			'target_landmarks': target_landmarks
		}


def create_lora_dataset(
	dataset_dir: str,
	source_profile_id: str,
	max_frames: int = 1000,
	frame_size: int = 128
) -> Optional[LoRAPairedDataset]:
	"""
	Helper function to create a LoRA dataset from a profile ID.

	Args:
		dataset_dir: Directory with extracted frames
		source_profile_id: ID of the source identity profile
		max_frames: Maximum frames to use
		frame_size: Target frame size

	Returns:
		LoRAPairedDataset instance or None if profile not found
	"""
	from watserface.identity_profile import get_identity_manager

	# Load source profile
	manager = get_identity_manager()
	profile = manager.source_intelligence.load_profile(source_profile_id)

	if not profile:
		logger.error(f"Source profile '{source_profile_id}' not found", __name__)
		return None

	if not profile.embedding_mean:
		logger.error(f"Source profile '{source_profile_id}' has no embeddings", __name__)
		return None

	# Convert embedding to numpy
	source_embedding = np.array(profile.embedding_mean, dtype=np.float32)

	# Create dataset
	dataset = LoRAPairedDataset(
		dataset_dir=dataset_dir,
		source_embedding=source_embedding,
		max_frames=max_frames,
		frame_size=frame_size
	)

	logger.info(f"Created LoRA dataset: {len(dataset)} paired samples", __name__)
	return dataset


# Example usage
if __name__ == "__main__":
	# Test dataset creation
	dataset_dir = ".jobs/training_dataset_lora"
	source_embedding = np.random.randn(512).astype(np.float32)

	dataset = LoRAPairedDataset(
		dataset_dir=dataset_dir,
		source_embedding=source_embedding,
		max_frames=100
	)

	# Test data loading
	source_emb, target_frame, landmarks = dataset[0]
	print(f"Source embedding shape: {source_emb.shape}")
	print(f"Target frame shape: {target_frame.shape}")
	print(f"Landmarks shape: {landmarks.shape}")

	# Test batch collation
	from torch.utils.data import DataLoader
	collator = LoRACollator()
	loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collator)

	batch = next(iter(loader))
	print(f"\nBatch shapes:")
	print(f"  Source embeddings: {batch['source_embeddings'].shape}")
	print(f"  Target frames: {batch['target_frames'].shape}")
	print(f"  Landmarks: {batch['target_landmarks'].shape}")
