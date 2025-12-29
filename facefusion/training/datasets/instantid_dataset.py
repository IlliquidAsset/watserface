"""
InstantID Dataset for Identity Training.

Pre-computes InsightFace embeddings and loads MediaPipe landmarks.
"""

from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2

from facefusion.training.dataset_extractor import load_dataset_manifest, load_sample
from facefusion.face_analyser import get_many_faces
from facefusion.vision import read_static_image


class InstantIDDataset(Dataset):
	"""
	Dataset for InstantID Identity Training.

	Pre-computes face embeddings and loads landmarks for each sample.

	Args:
		dataset_dir: Path to dataset directory (with images/ and landmarks/ subdirs)
		image_size: Image size for training (default 256)
		cache_embeddings: If True, pre-compute and cache all embeddings (faster but uses memory)
	"""

	def __init__(
		self,
		dataset_dir: str,
		image_size: int = 256,
		cache_embeddings: bool = True
	):
		self.dataset_dir = Path(dataset_dir)
		self.image_size = image_size
		self.cache_embeddings = cache_embeddings

		# Load manifest
		self.manifest = load_dataset_manifest(str(self.dataset_dir))
		self.samples = self.manifest['samples']

		# Pre-compute embeddings if caching enabled
		if self.cache_embeddings:
			print("Pre-computing face embeddings...")
			self._embedding_cache = {}
			self._precompute_embeddings()
		else:
			self._embedding_cache = None

	def _precompute_embeddings(self):
		"""Pre-compute all face embeddings for faster training."""
		for idx, sample in enumerate(self.samples):
			if idx % 10 == 0:
				print(f"  Computing embeddings: {idx}/{len(self.samples)}")

			# Load image
			image_path = self.dataset_dir / sample['image_path']
			image = read_static_image(str(image_path))

			if image is None:
				print(f"Warning: Could not load image {image_path}")
				self._embedding_cache[idx] = None
				continue

			# Extract embedding
			embedding = self._extract_embedding(image)
			self._embedding_cache[idx] = embedding

		print(f"✅ Pre-computed {len(self._embedding_cache)} embeddings")

	def _extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
		"""
		Extract InsightFace embedding from image.

		Args:
			image: (H, W, 3) BGR image

		Returns:
			(512,) embedding or None if no face detected
		"""
		# Detect face
		faces = get_many_faces([image])

		if not faces or len(faces) == 0 or len(faces[0]) == 0:
			return None

		face = faces[0][0]  # Use first detected face

		# Extract embedding
		if hasattr(face, 'embedding') and face.embedding is not None:
			return face.embedding
		else:
			return None

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		sample = self.samples[idx]

		# Load image
		image_path = self.dataset_dir / sample['image_path']
		image = cv2.imread(str(image_path))
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		# Resize image
		image = cv2.resize(image, (self.image_size, self.image_size))

		# Load landmarks
		landmarks_path = self.dataset_dir / sample['landmarks_path']
		landmarks = np.load(str(landmarks_path))  # (478, 2)

		# Normalize landmarks to [0, 1]
		# Assumes landmarks are in image coordinates
		h, w = image.shape[:2]
		landmarks_normalized = landmarks.copy()
		landmarks_normalized[:, 0] /= w
		landmarks_normalized[:, 1] /= h

		# Get or compute embedding
		if self._embedding_cache is not None:
			# Use cached embedding
			embedding = self._embedding_cache.get(idx)
		else:
			# Compute on-the-fly
			image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			embedding = self._extract_embedding(image_bgr)

		if embedding is None:
			# Fallback: return zero embedding if extraction failed
			embedding = np.zeros(512, dtype=np.float32)
			print(f"Warning: No embedding for sample {idx}, using zero vector")

		# Convert to tensors
		image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (3, H, W), [0, 1]
		embedding_tensor = torch.from_numpy(embedding).float()  # (512,)
		landmarks_tensor = torch.from_numpy(landmarks_normalized).float()  # (478, 2)

		return {
			'image': image_tensor,
			'embedding': embedding_tensor,
			'landmarks': landmarks_tensor,
			'sample_name': sample['name']
		}

	def get_embedding_stats(self) -> dict:
		"""
		Get statistics about embeddings in dataset.

		Returns:
			{
				'total_samples': int,
				'valid_embeddings': int,
				'missing_embeddings': int,
				'embedding_coverage': float (percentage)
			}
		"""
		if self._embedding_cache is None:
			return {
				'total_samples': len(self.samples),
				'valid_embeddings': -1,
				'missing_embeddings': -1,
				'embedding_coverage': -1
			}

		total = len(self._embedding_cache)
		valid = sum(1 for e in self._embedding_cache.values() if e is not None)
		missing = total - valid
		coverage = (valid / total * 100) if total > 0 else 0

		return {
			'total_samples': total,
			'valid_embeddings': valid,
			'missing_embeddings': missing,
			'embedding_coverage': coverage
		}


if __name__ == '__main__':
	# Test dataset loading
	import sys

	if len(sys.argv) < 2:
		print("Usage: python instantid_dataset.py <dataset_dir>")
		sys.exit(1)

	dataset_dir = sys.argv[1]

	# Load dataset
	dataset = InstantIDDataset(dataset_dir, cache_embeddings=True)
	print(f"\nDataset loaded: {len(dataset)} samples")

	# Get embedding stats
	stats = dataset.get_embedding_stats()
	print(f"\nEmbedding Statistics:")
	print(f"  Total samples: {stats['total_samples']}")
	print(f"  Valid embeddings: {stats['valid_embeddings']}")
	print(f"  Missing embeddings: {stats['missing_embeddings']}")
	print(f"  Coverage: {stats['embedding_coverage']:.1f}%")

	# Test first sample
	sample = dataset[0]
	print(f"\nFirst sample:")
	print(f"  Image shape: {sample['image'].shape}")
	print(f"  Embedding shape: {sample['embedding'].shape}")
	print(f"  Landmarks shape: {sample['landmarks'].shape}")
	print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
	print(f"  Embedding range: [{sample['embedding'].min():.3f}, {sample['embedding'].max():.3f}]")
	print(f"  Landmarks range: [{sample['landmarks'].min():.3f}, {sample['landmarks'].max():.3f}]")
	print(f"  Sample name: {sample['sample_name']}")
