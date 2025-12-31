"""
Quick utility to export LoRA checkpoint to ONNX
"""
import os
import sys
import torch
import shutil

def export_lora_checkpoint(checkpoint_path: str, model_name: str):
	"""Export a LoRA checkpoint to ONNX"""
	print(f"Loading checkpoint: {checkpoint_path}")

	# Load checkpoint
	checkpoint = torch.load(checkpoint_path, map_location='cpu')

	if not isinstance(checkpoint, dict):
		print("Error: Checkpoint is not a dictionary")
		return False

	lora_rank = checkpoint.get('lora_rank', 16)
	print(f"LoRA Rank: {lora_rank}")
	print(f"Epoch: {checkpoint.get('epoch', '?')}")
	print(f"Loss: {checkpoint.get('loss', '?')}")

	# Create model with LoRA
	from watserface.training.train_instantid import IdentityGenerator
	from watserface.training.models.lora_adapter import LoRAWrapper

	base_model = IdentityGenerator()
	model = LoRAWrapper.add_lora_to_model(
		base_model,
		target_modules=None,
		rank=lora_rank,
		alpha=float(lora_rank),
		dropout=0.1
	)

	# Load LoRA state
	LoRAWrapper.load_lora_state_dict(model, checkpoint.get('lora_state', {}))
	model.eval()

	print("Model loaded successfully")

	# Export to ONNX
	output_path = f".assets/models/trained/{model_name}_lora.onnx"
	os.makedirs(os.path.dirname(output_path), exist_ok=True)

	dummy_target = torch.randn(1, 3, 128, 128)
	dummy_source = torch.randn(1, 512)

	print(f"Exporting to {output_path}...")
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

	print(f"✅ Exported to {output_path}")

	# Copy checkpoint
	checkpoint_dst = f".assets/models/trained/{model_name}_lora.pth"
	shutil.copy(checkpoint_path, checkpoint_dst)
	print(f"✅ Copied checkpoint to {checkpoint_dst}")

	# Generate hash
	import zlib
	with open(output_path, 'rb') as f:
		model_content = f.read()
	model_hash = format(zlib.crc32(model_content), '08x')
	hash_path = f".assets/models/trained/{model_name}_lora.hash"
	with open(hash_path, 'w') as f:
		f.write(model_hash)
	print(f"✅ Generated hash: {hash_path}")

	return True

if __name__ == "__main__":
	checkpoint_path = ".jobs/training_dataset_lora/s2b_lora.pth"
	model_name = "s2b"

	if export_lora_checkpoint(checkpoint_path, model_name):
		print("\n✅ Export complete! s2b_lora should now appear in Swap tab.")
	else:
		print("\n❌ Export failed")
