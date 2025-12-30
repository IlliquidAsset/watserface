"""
InstantID Adapter Architecture.

Combines face embeddings (512-dim from InsightFace) with pose landmarks (478×2 from MediaPipe)
to create identity-preserving representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstantIDAdapter(nn.Module):
	"""
	InstantID Identity Encoder.

	Combines identity embedding + pose landmarks into a unified representation.

	Architecture:
	- Face Encoder: Linear(512 → 1024) for InsightFace embeddings
	- Landmark Encoder: Linear(956 → 1024) for flattened 478×2 landmarks
	- Fusion: Concatenate + Project (2048 → 768) for CLIP dimension

	Input:
		face_embedding: (B, 512) InsightFace embedding
		landmarks_478: (B, 478, 2) MediaPipe landmarks

	Output:
		identity_encoding: (B, 768) Fused identity+pose encoding
	"""

	def __init__(
		self,
		embedding_dim: int = 512,
		landmark_dim: int = 956,  # 478 * 2
		hidden_dim: int = 1024,
		output_dim: int = 768,  # CLIP dimension
		dropout: float = 0.1
	):
		super().__init__()

		self.embedding_dim = embedding_dim
		self.landmark_dim = landmark_dim
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim

		# Face embedding encoder
		self.face_encoder = nn.Sequential(
			nn.Linear(embedding_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(inplace=True)
		)

		# Landmark encoder
		self.landmark_encoder = nn.Sequential(
			nn.Linear(landmark_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, hidden_dim),
			nn.LayerNorm(hidden_dim),
			nn.ReLU(inplace=True)
		)

		# Fusion layer
		self.fusion = nn.Sequential(
			nn.Linear(hidden_dim * 2, output_dim),
			nn.LayerNorm(output_dim),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(output_dim, output_dim)
		)

	def forward(self, face_embedding: torch.Tensor, landmarks_478: torch.Tensor) -> torch.Tensor:
		"""
		Forward pass.

		Args:
			face_embedding: (B, 512) InsightFace embedding
			landmarks_478: (B, 478, 2) MediaPipe landmarks

		Returns:
			identity_encoding: (B, 768) Fused encoding
		"""
		batch_size = face_embedding.shape[0]

		# Encode face embedding
		face_features = self.face_encoder(face_embedding)  # (B, 1024)

		# Flatten and encode landmarks
		landmarks_flat = landmarks_478.view(batch_size, -1)  # (B, 956)
		landmark_features = self.landmark_encoder(landmarks_flat)  # (B, 1024)

		# Concatenate and fuse
		combined = torch.cat([face_features, landmark_features], dim=1)  # (B, 2048)
		identity_encoding = self.fusion(combined)  # (B, 768)

		return identity_encoding

	def encode_identity(self, face_embedding: torch.Tensor, landmarks_478: torch.Tensor) -> torch.Tensor:
		"""Alias for forward pass (for compatibility)."""
		return self.forward(face_embedding, landmarks_478)


class InstantIDLoss(nn.Module):
	"""
	Loss function for InstantID training.

	Combines:
	1. Identity loss: MSE between encoded and target embeddings
	2. Landmark consistency loss: Ensure landmarks are preserved
	"""

	def __init__(
		self,
		identity_weight: float = 1.0,
		landmark_weight: float = 0.1
	):
		super().__init__()
		self.identity_weight = identity_weight
		self.landmark_weight = landmark_weight

	def forward(
		self,
		encoded: torch.Tensor,
		target_embedding: torch.Tensor,
		input_landmarks: torch.Tensor,
		reconstructed_landmarks: torch.Tensor = None
	) -> dict:
		"""
		Compute combined loss.

		Args:
			encoded: (B, 768) Encoded identity
			target_embedding: (B, 768) Target embedding (truncated to match output_dim)
			input_landmarks: (B, 478, 2) Input landmarks
			reconstructed_landmarks: (B, 478, 2) Reconstructed landmarks (optional)

		Returns:
			{
				'total_loss': Combined loss,
				'identity_loss': Identity matching loss,
				'landmark_loss': Landmark consistency loss (if applicable)
			}
		"""
		# Identity loss (MSE)
		identity_loss = F.mse_loss(encoded, target_embedding[:, :encoded.shape[1]])

		# Landmark consistency loss (if provided)
		if reconstructed_landmarks is not None:
			landmark_loss = F.mse_loss(reconstructed_landmarks, input_landmarks)
		else:
			landmark_loss = torch.tensor(0.0, device=encoded.device)

		# Combined loss
		total_loss = (
			self.identity_weight * identity_loss +
			self.landmark_weight * landmark_loss
		)

		return {
			'total_loss': total_loss,
			'identity_loss': identity_loss,
			'landmark_loss': landmark_loss
		}


def test_instantid_adapter():
	"""Test InstantIDAdapter architecture."""
	model = InstantIDAdapter(
		embedding_dim=512,
		landmark_dim=956,
		hidden_dim=1024,
		output_dim=768
	)

	# Test forward pass
	batch_size = 4
	face_embedding = torch.randn(batch_size, 512)
	landmarks_478 = torch.randn(batch_size, 478, 2)

	with torch.no_grad():
		output = model(face_embedding, landmarks_478)

	print(f"Face embedding shape: {face_embedding.shape}")
	print(f"Landmarks shape: {landmarks_478.shape}")
	print(f"Output shape: {output.shape}")
	print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

	# Count parameters
	total_params = sum(p.numel() for p in model.parameters())
	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

	print(f"Total parameters: {total_params:,}")
	print(f"Trainable parameters: {trainable_params:,}")

	assert output.shape == (batch_size, 768), "Output shape mismatch!"

	# Test loss
	loss_fn = InstantIDLoss()
	target_embedding = torch.randn(batch_size, 768)
	loss_dict = loss_fn(output, target_embedding, landmarks_478)

	print(f"\nLoss test:")
	print(f"  Total loss: {loss_dict['total_loss']:.4f}")
	print(f"  Identity loss: {loss_dict['identity_loss']:.4f}")
	print(f"  Landmark loss: {loss_dict['landmark_loss']:.4f}")

	print("\n✅ InstantIDAdapter test passed!")


if __name__ == '__main__':
	test_instantid_adapter()
