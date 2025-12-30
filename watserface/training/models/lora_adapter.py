"""
LoRA (Low-Rank Adaptation) Layer Implementation

LoRA adds trainable low-rank matrices to frozen model weights,
enabling efficient fine-tuning with minimal parameters.

Architecture:
    W_frozen + (B × A) where:
    - W_frozen: Original model weights (frozen)
    - A: Down-projection matrix (rank reduction)
    - B: Up-projection matrix (rank expansion)
    - rank << original dimensions (e.g., 16 vs 512)

Memory: ~0.1% of full model parameters
Quality: 95%+ of full fine-tuning performance
"""

import torch
import torch.nn as nn
from typing import Optional


class LoRALayer(nn.Module):
	"""
	LoRA layer that wraps an existing linear/conv layer.

	Implements: output = W_frozen(x) + alpha * B(A(x))
	"""

	def __init__(
		self,
		original_layer: nn.Module,
		rank: int = 16,
		alpha: float = 16.0,
		dropout: float = 0.0
	):
		"""
		Args:
			original_layer: The frozen layer to adapt (Linear or Conv2d)
			rank: LoRA rank (lower = fewer parameters, faster)
			alpha: Scaling factor for LoRA output (typically = rank)
			dropout: Dropout probability for LoRA path
		"""
		super().__init__()

		self.original_layer = original_layer
		self.rank = rank
		self.alpha = alpha

		# Freeze original layer
		for param in self.original_layer.parameters():
			param.requires_grad = False

		# Determine layer type and dimensions
		if isinstance(original_layer, nn.Linear):
			in_features = original_layer.in_features
			out_features = original_layer.out_features

			# LoRA matrices: A (down), B (up)
			self.lora_A = nn.Linear(in_features, rank, bias=False)
			self.lora_B = nn.Linear(rank, out_features, bias=False)

		elif isinstance(original_layer, nn.Conv2d):
			in_channels = original_layer.in_channels
			out_channels = original_layer.out_channels
			kernel_size = original_layer.kernel_size

			# LoRA for Conv2d
			self.lora_A = nn.Conv2d(
				in_channels, rank,
				kernel_size=kernel_size,
				stride=original_layer.stride,
				padding=original_layer.padding,
				bias=False
			)
			self.lora_B = nn.Conv2d(
				rank, out_channels,
				kernel_size=1,  # 1x1 conv for channel expansion
				bias=False
			)
		else:
			raise ValueError(f"LoRA only supports Linear and Conv2d layers, got {type(original_layer)}")

		# Initialize LoRA weights
		nn.init.kaiming_uniform_(self.lora_A.weight, a=1)  # Small random init
		nn.init.zeros_(self.lora_B.weight)  # Zero init (LoRA starts as identity)

		# Optional dropout
		self.dropout = nn.Dropout(dropout) if dropout > 0 else None

		# Scaling factor
		self.scaling = alpha / rank

	def forward(self, x):
		"""
		Forward pass: frozen output + scaled LoRA output
		"""
		# Frozen base model output
		base_output = self.original_layer(x)

		# LoRA path: x -> A -> dropout -> B -> scale
		lora_output = self.lora_A(x)
		if self.dropout is not None:
			lora_output = self.dropout(lora_output)
		lora_output = self.lora_B(lora_output)
		lora_output = lora_output * self.scaling

		return base_output + lora_output

	def merge_weights(self):
		"""
		Merge LoRA weights back into original layer (for inference).
		Returns merged weight tensor.
		"""
		with torch.no_grad():
			if isinstance(self.original_layer, nn.Linear):
				# W_merged = W_frozen + alpha * (B @ A)
				lora_weight = self.lora_B.weight @ self.lora_A.weight
				merged_weight = self.original_layer.weight + (lora_weight * self.scaling)
				return merged_weight

			elif isinstance(self.original_layer, nn.Conv2d):
				# For Conv2d, reshape and merge
				# This is more complex - for now return decomposed form
				# Full implementation would require careful reshaping
				return None  # Placeholder - keep decomposed for ONNX export


class LoRAWrapper:
	"""
	Utility to wrap a model with LoRA layers.
	"""

	@staticmethod
	def add_lora_to_model(
		model: nn.Module,
		target_modules: list = None,
		rank: int = 16,
		alpha: float = 16.0,
		dropout: float = 0.0
	) -> nn.Module:
		"""
		Add LoRA layers to specified modules in a model.

		Args:
			model: Base model to adapt
			target_modules: List of module names to add LoRA to
			                (e.g., ['enc.0', 'enc.2', 'res_blocks.0.conv.0'])
			rank: LoRA rank
			alpha: LoRA scaling
			dropout: LoRA dropout

		Returns:
			Modified model with LoRA layers
		"""
		if target_modules is None:
			# Default: Add LoRA to all Conv2d and Linear layers
			target_modules = []
			for name, module in model.named_modules():
				if isinstance(module, (nn.Linear, nn.Conv2d)):
					target_modules.append(name)

		# Replace target modules with LoRA-wrapped versions
		for target_name in target_modules:
			# Navigate to parent module
			parent_name, child_name = target_name.rsplit('.', 1) if '.' in target_name else ('', target_name)
			parent = model if not parent_name else model.get_submodule(parent_name)

			# Get original layer
			original_layer = getattr(parent, child_name)

			# Wrap with LoRA
			lora_layer = LoRALayer(original_layer, rank=rank, alpha=alpha, dropout=dropout)

			# Replace
			setattr(parent, child_name, lora_layer)

		return model

	@staticmethod
	def count_lora_parameters(model: nn.Module) -> tuple:
		"""
		Count trainable LoRA parameters vs total model parameters.

		Returns:
			(trainable_params, total_params, percentage)
		"""
		trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
		total = sum(p.numel() for p in model.parameters())
		percentage = (trainable / total * 100) if total > 0 else 0

		return trainable, total, percentage

	@staticmethod
	def extract_lora_state_dict(model: nn.Module) -> dict:
		"""
		Extract only LoRA parameters from model.
		Useful for saving lightweight checkpoints.
		"""
		lora_state = {}
		for name, module in model.named_modules():
			if isinstance(module, LoRALayer):
				lora_state[f"{name}.lora_A"] = module.lora_A.state_dict()
				lora_state[f"{name}.lora_B"] = module.lora_B.state_dict()
		return lora_state

	@staticmethod
	def load_lora_state_dict(model: nn.Module, lora_state: dict):
		"""
		Load LoRA parameters into model.
		"""
		for name, module in model.named_modules():
			if isinstance(module, LoRALayer):
				if f"{name}.lora_A" in lora_state:
					module.lora_A.load_state_dict(lora_state[f"{name}.lora_A"])
				if f"{name}.lora_B" in lora_state:
					module.lora_B.load_state_dict(lora_state[f"{name}.lora_B"])


# Example usage:
if __name__ == "__main__":
	# Create a simple base model
	class SimpleModel(nn.Module):
		def __init__(self):
			super().__init__()
			self.fc1 = nn.Linear(512, 256)
			self.fc2 = nn.Linear(256, 128)

	base_model = SimpleModel()
	print(f"Base model parameters: {sum(p.numel() for p in base_model.parameters())}")

	# Add LoRA
	lora_model = LoRAWrapper.add_lora_to_model(
		base_model,
		target_modules=['fc1', 'fc2'],
		rank=16,
		alpha=16.0
	)

	trainable, total, pct = LoRAWrapper.count_lora_parameters(lora_model)
	print(f"Trainable LoRA params: {trainable} / {total} ({pct:.2f}%)")

	# Forward pass
	x = torch.randn(1, 512)
	out = lora_model.fc1(x)
	print(f"Output shape: {out.shape}")
