import os
import sys
from watserface import state_manager
from watserface.identity_profile import get_source_intelligence, IdentityProfileConfig
from watserface.training.train_lora import train_lora_model
from watserface import logger

def main():
    # Initialize State Manager
    state_manager.init_item('face_detector_model', 'yolo_face')
    state_manager.init_item('face_detector_size', '640x640')
    state_manager.init_item('face_detector_angles', [0])
    state_manager.init_item('face_detector_score', 0.5)
    state_manager.init_item('face_landmarker_model', '2dfan4')
    state_manager.init_item('face_landmarker_score', 0.5)
    state_manager.init_item('download_providers', ['github'])
    state_manager.init_item('execution_providers', ['cpu'])
    state_manager.init_item('execution_device_id', '0')

    # 1. Create Identity Profile for Source (Sam)
    source_path = "/Users/kendrick/Documents/FS Source/Sam_Generated.png"
    if not os.path.exists(source_path):
        print(f"Error: Source image not found at {source_path}")
        return

    print(f"Creating profile for {source_path}...")
    
    # Use custom config to allow "blurry" AI images
    config = IdentityProfileConfig(blur_var_min=10.0) 
    intelligence = get_source_intelligence()
    intelligence.config = config # Override config
    
    # Force creation even if single image, to get an ID
    try:
        profile = intelligence.create_identity_profile(
            [source_path], 
            profile_name="Sam_Source_Profile", 
            save_persistent=True
        )
    except Exception as e:
        print(f"Failed to create profile: {e}")
        return

    source_profile_id = profile.id
    print(f"Created Profile ID: {source_profile_id}")

    # 2. Train LoRA using the Face Set (Target Frames)
    # Note: As analyzed, this technically trains the model to reconstruct the TARGET face
    # from the SOURCE embedding. This might be intended for 'Target-Aware' adapters
    # or specific re-lighting tasks.
    dataset_dir = "models/face_sets/faceset_512e84d4_1768337182/frames"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Face set directory not found at {dataset_dir}")
        return

    model_name = "sam_to_zbam_lora"
    
    print(f"Starting LoRA training: {model_name}")
    print(f"Dataset: {dataset_dir}")
    print(f"Source: {source_profile_id}")

    # Training parameters
    # Using small epochs for PoC
    iterator = train_lora_model(
        dataset_dir=dataset_dir,
        source_profile_id=source_profile_id,
        model_name=model_name,
        epochs=10, 
        batch_size=2, # Small batch for safety
        learning_rate=1e-4,
        lora_rank=4, # Lightweight
        save_interval=5
    )

    # Consuming the iterator to run training
    for status, telemetry in iterator:
        print(f"[Train] {status}")

if __name__ == "__main__":
    main()
