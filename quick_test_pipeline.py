"""
Quick Test Pipeline - Identity + LoRA Training + Swap
Skips UI, uses hardcoded paths, minimal epochs for testing
"""
import os
import sys
import time
from pathlib import Path

# Add watserface to path
sys.path.insert(0, str(Path(__file__).parent))

from watserface import identity_profile, logger, state_manager
from watserface.training import core as training_core
from watserface.filesystem import resolve_relative_path


def print_section(title: str):
	"""Print a section header"""
	print("\n" + "=" * 80)
	print(f"  {title}")
	print("=" * 80 + "\n")


def print_telemetry(data: dict, prefix=""):
	"""Print telemetry data in a formatted way"""
	for key, value in data.items():
		if isinstance(value, dict):
			print(f"{prefix}{key}:")
			print_telemetry(value, prefix + "  ")
		else:
			print(f"{prefix}{key}: {value}")


def test_identity_training(source_path: str, model_name: str, min_epochs: int = 30):
	"""Test identity training with minimal epochs"""
	print_section(f"PHASE 1: Identity Training - '{model_name}'")

	print(f"📁 Source: {source_path}")
	print(f"🔄 Epochs: {min_epochs} (minimal for testing)")
	print(f"🎯 Model: {model_name}")
	print()
	print(f"💡 Note: Training uses checkpoints!")
	print(f"   - Saves every 5-10 epochs automatically")
	print(f"   - If you quit (Ctrl+C), progress is saved")
	print(f"   - Rerun to resume from last checkpoint")
	print(f"   - Frames are sampled to max 1000 (not all frames)")
	print()

	start_time = time.time()
	final_status = None

	# Run identity training
	for status_data in training_core.start_identity_training(
		model_name=model_name,
		epochs=min_epochs,
		source_files=source_path
	):
		if isinstance(status_data, list) and len(status_data) >= 2:
			message, telemetry = status_data[0], status_data[1]

			# Print status updates
			print(f"📊 {message}")

			if isinstance(telemetry, dict) and telemetry:
				print_telemetry(telemetry, "   ")
				print()

			final_status = (message, telemetry)
		else:
			print(f"Status: {status_data}")
			final_status = status_data

	elapsed = time.time() - start_time

	print(f"\n⏱️  Total Training Time: {elapsed:.1f}s")

	# Verify profile was created
	manager = identity_profile.get_identity_manager()
	profile = manager.source_intelligence.load_profile(model_name.lower().replace(' ', '_'))

	if profile:
		print(f"\n✅ Identity Profile Created Successfully!")
		print(f"   Profile ID: {profile.id}")
		print(f"   Name: {profile.name}")
		print(f"   Embeddings: {len(profile.embedding_mean)}")
		print(f"   Quality Stats:")
		print_telemetry(profile.quality_stats, "      ")
		return profile.id
	else:
		print(f"\n❌ Identity Profile NOT Created")
		return None


def test_lora_training(profile_id: str, target_path: str, model_name: str, min_epochs: int = 50):
	"""Test LoRA training (stub for now)"""
	print_section(f"PHASE 2: LoRA Training - '{model_name}'")

	print(f"👤 Source Profile: {profile_id}")
	print(f"🎯 Target: {target_path}")
	print(f"🔄 Epochs: {min_epochs} (minimal for testing)")
	print(f"📦 LoRA Rank: 16")
	print(f"📚 Batch Size: 4")
	print(f"📈 Learning Rate: 0.0001")
	print()

	print("🚧 LoRA training not yet implemented (Phase 2)")
	print("   This will be implemented next with:")
	print("   - LoRA adapter architecture")
	print("   - Paired dataset loader")
	print("   - Training loop with ONNX export")
	print()

	return None


def test_face_swap(source_profile_id: str, target_path: str, output_path: str, duration_sec: int = 10):
	"""Test face swap with trained model"""
	print_section(f"PHASE 3: Face Swap - Output {duration_sec}s @ 30fps")

	print(f"👤 Source Profile: {source_profile_id}")
	print(f"🎯 Target: {target_path}")
	print(f"📤 Output: {output_path}")
	print(f"⏱️  Duration: {duration_sec}s (300 frames @ 30fps)")
	print()

	print("🚧 Face swap integration not yet implemented")
	print("   This will use the trained identity profile for swapping")
	print()

	return None


def main():
	"""Main test pipeline"""
	print_section("Quick Test Pipeline - Identity + LoRA + Swap")

	# Configuration
	SOURCE_PATH = ".assets/examples/source.jpg"  # Change to your test source
	TARGET_PATH = ".assets/examples/target-1080p.mp4"  # Change to your test target
	OUTPUT_PATH = ".assets/output/test_quick_pipeline.mp4"

	IDENTITY_NAME = "test_identity"
	LORA_MODEL_NAME = "test_lora_model"

	MIN_IDENTITY_EPOCHS = 30  # Minimum for reasonable identity quality
	MIN_LORA_EPOCHS = 50  # Minimum for reasonable LoRA quality
	OUTPUT_DURATION_SEC = 10  # Output only 10 seconds

	# Verify files exist
	if not os.path.exists(SOURCE_PATH):
		print(f"❌ Source file not found: {SOURCE_PATH}")
		print(f"   Please update SOURCE_PATH in this script to point to your test source")
		return

	if not os.path.exists(TARGET_PATH):
		print(f"❌ Target file not found: {TARGET_PATH}")
		print(f"   Please update TARGET_PATH in this script to point to your test target")
		return

	print(f"📂 Test Configuration:")
	print(f"   Source: {SOURCE_PATH}")
	print(f"   Target: {TARGET_PATH}")
	print(f"   Output: {OUTPUT_PATH}")
	print(f"   Identity Name: {IDENTITY_NAME}")
	print(f"   LoRA Model Name: {LORA_MODEL_NAME}")
	print(f"   Identity Epochs: {MIN_IDENTITY_EPOCHS}")
	print(f"   LoRA Epochs: {MIN_LORA_EPOCHS}")
	print(f"   Output Duration: {OUTPUT_DURATION_SEC}s")
	print()

	# Initialize state manager (minimal setup)
	state_manager.init_item('jobs_path', '.jobs')
	state_manager.init_item('output_path', OUTPUT_PATH)
	state_manager.init_item('output_video_fps', 30)

	# Phase 1: Train Identity
	profile_id = test_identity_training(SOURCE_PATH, IDENTITY_NAME, MIN_IDENTITY_EPOCHS)

	if not profile_id:
		print("\n❌ Identity training failed. Aborting pipeline.")
		return

	# Phase 2: Train LoRA Model (stub for now)
	lora_model_path = test_lora_training(profile_id, TARGET_PATH, LORA_MODEL_NAME, MIN_LORA_EPOCHS)

	# Phase 3: Face Swap with trained model (stub for now)
	test_face_swap(profile_id, TARGET_PATH, OUTPUT_PATH, OUTPUT_DURATION_SEC)

	print_section("Pipeline Complete!")
	print(f"✅ Identity Profile: {profile_id}")
	print(f"🚧 LoRA Model: Not yet implemented")
	print(f"🚧 Face Swap Output: Not yet implemented")
	print()
	print(f"Next steps:")
	print(f"1. Implement LoRA training system (Phase 2)")
	print(f"2. Integrate face swap with identity profiles")
	print(f"3. Add context-aware blending")
	print(f"4. Add InstantID refinement")
	print()


if __name__ == "__main__":
	main()
