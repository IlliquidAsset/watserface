"""
Device Utilities for Training.

Auto-detection and optimization for MPS (Apple Silicon), CUDA, and CPU.
"""

import torch
import torch.nn as nn


def get_optimal_device() -> str:
	"""
	Auto-detect optimal training device.

	Priority:
	1. Apple Silicon MPS (if available)
	2. NVIDIA CUDA (if available)
	3. CPU (fallback)

	Returns:
		Device string: 'mps', 'cuda', or 'cpu'
	"""
	# Check for Apple Silicon MPS
	if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
		print("✅ Apple Silicon MPS detected")
		return 'mps'

	# Check for NVIDIA CUDA
	if torch.cuda.is_available():
		device_name = torch.cuda.get_device_name(0)
		print(f"✅ CUDA device detected: {device_name}")
		return 'cuda'

	# Fallback to CPU
	print("⚠️  No GPU detected, using CPU (training will be slower)")
	return 'cpu'


def optimize_for_device(model: nn.Module, device: str) -> nn.Module:
	"""
	Optimize model for specific device.

	Args:
		model: PyTorch model
		device: Device string ('mps', 'cuda', or 'cpu')

	Returns:
		Optimized model
	"""
	if device == 'mps':
		# MPS optimization
		# Note: MPS doesn't fully support float16 yet, use float32
		model = model.float()
		torch.set_float32_matmul_precision('high')
		print("🔧 Optimized for Apple Silicon MPS (float32)")

	elif device == 'cuda':
		# CUDA optimization
		# Can use mixed precision for faster training
		try:
			model = model.half()  # Use float16 for memory efficiency
			print("🔧 Optimized for CUDA with float16")
		except Exception as e:
			print(f"⚠️  Could not use float16, falling back to float32: {e}")
			model = model.float()

	else:  # CPU
		# CPU optimization
		model = model.float()
		# Set number of threads for CPU inference
		torch.set_num_threads(torch.get_num_threads())
		print(f"🔧 Optimized for CPU ({torch.get_num_threads()} threads)")

	return model


def get_device_info() -> dict:
	"""
	Get detailed device information.

	Returns:
		{
			'device': str ('mps', 'cuda', or 'cpu'),
			'device_name': str,
			'memory_total': int (bytes, if applicable),
			'memory_available': int (bytes, if applicable),
			'supports_fp16': bool
		}
	"""
	device = get_optimal_device()

	info = {
		'device': device,
		'device_name': 'Unknown',
		'memory_total': 0,
		'memory_available': 0,
		'supports_fp16': False
	}

	if device == 'mps':
		info['device_name'] = 'Apple Silicon MPS'
		info['supports_fp16'] = False  # MPS has limited fp16 support

	elif device == 'cuda':
		info['device_name'] = torch.cuda.get_device_name(0)
		info['memory_total'] = torch.cuda.get_device_properties(0).total_memory
		info['memory_available'] = info['memory_total'] - torch.cuda.memory_allocated(0)
		info['supports_fp16'] = True

	else:  # CPU
		import psutil
		info['device_name'] = f"CPU ({torch.get_num_threads()} threads)"
		info['memory_total'] = psutil.virtual_memory().total
		info['memory_available'] = psutil.virtual_memory().available
		info['supports_fp16'] = False

	return info


def test_device():
	"""Test device detection and optimization."""
	print("=" * 60)
	print("Device Detection Test")
	print("=" * 60)

	# Get device info
	info = get_device_info()

	print(f"\nDevice: {info['device']}")
	print(f"Name: {info['device_name']}")

	if info['memory_total'] > 0:
		total_gb = info['memory_total'] / (1024 ** 3)
		available_gb = info['memory_available'] / (1024 ** 3)
		print(f"Memory: {available_gb:.1f} GB / {total_gb:.1f} GB available")

	print(f"FP16 Support: {info['supports_fp16']}")

	# Test model creation and optimization
	print("\n" + "=" * 60)
	print("Model Optimization Test")
	print("=" * 60)

	class DummyModel(nn.Module):
		def __init__(self):
			super().__init__()
			self.linear = nn.Linear(100, 10)

		def forward(self, x):
			return self.linear(x)

	model = DummyModel()
	device = info['device']

	print(f"\nOptimizing model for {device}...")
	model = optimize_for_device(model, device)
	model = model.to(device)

	# Test forward pass
	dummy_input = torch.randn(4, 100).to(device)
	if device == 'cuda' and info['supports_fp16']:
		dummy_input = dummy_input.half()

	with torch.no_grad():
		output = model(dummy_input)

	print(f"✅ Forward pass successful!")
	print(f"   Input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")
	print(f"   Output shape: {output.shape}, dtype: {output.dtype}")

	print("\n" + "=" * 60)
	print("✅ All tests passed!")
	print("=" * 60)


if __name__ == '__main__':
	test_device()
