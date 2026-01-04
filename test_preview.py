#!/usr/bin/env python3
"""
Preview Test Script for s2b_32_lora

Tests the face swapping preview functionality with s2b_32_lora model
and saves the result to previewtest/ directory.
"""
import os
import sys
import random
import numpy
from datetime import datetime

# Add watserface to path
sys.path.insert(0, os.path.dirname(__file__))

from watserface import state_manager, logger
from watserface.logger import init as init_logger
from watserface.identity_profile import get_identity_manager
from watserface.processors.modules import face_swapper
from watserface.types import Face
from watserface.vision import read_image, write_image


def create_preview_test():
    """Run preview test with s2b_32_lora"""
    # Initialize logger
    init_logger('info')

    # Create output directory
    output_dir = 'previewtest'
    os.makedirs(output_dir, exist_ok=True)

    # Configuration
    lora_model = 's2b_32_lora'
    source_profile_id = 'sam_ident'
    frames_dir = '.jobs/training_dataset_lora'

    print(f"\n{'='*60}")
    print(f"PREVIEW TEST: {lora_model}")
    print(f"{'='*60}\n")

    # Find available frames
    frame_files = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    if not frame_files:
        print(f"ERROR: No frames found in {frames_dir}")
        return False

    # Select random frame
    random_frame = random.choice(frame_files)
    frame_path = os.path.join(frames_dir, random_frame)
    print(f"1. Selected random frame: {random_frame}")

    # Load source identity
    print(f"2. Loading source identity: {source_profile_id}")
    try:
        manager = get_identity_manager()
        profile = manager.source_intelligence.load_profile(source_profile_id)

        if not profile or not profile.embedding_mean:
            print(f"ERROR: Failed to load profile {source_profile_id}")
            return False

        embedding_mean = numpy.array(profile.embedding_mean, dtype=numpy.float64)
        source_face = Face(
            bounding_box=None,
            score_set=None,
            landmark_set=None,
            angle=None,
            embedding=embedding_mean,
            normed_embedding=embedding_mean,
            gender=None,
            age=None,
            race=None
        )
        print(f"   ✓ Profile loaded: {profile.name}, embedding_dim={len(profile.embedding_mean)}")

    except Exception as e:
        print(f"ERROR loading profile: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Configure state for face swapper
    print(f"3. Configuring face swapper with model: {lora_model}")
    # Face swapper settings
    state_manager.init_item('face_swapper_model', lora_model)
    state_manager.init_item('face_selector_mode', 'one')
    state_manager.init_item('face_swapper_pixel_boost', '128x128')
    state_manager.init_item('face_mask_types', ['box', 'occlusion'])
    state_manager.init_item('face_mask_blur', 0.3)
    state_manager.init_item('face_mask_padding', (0, 0, 0, 0))
    state_manager.init_item('processors', ['face_swapper'])

    # Face detection settings
    state_manager.init_item('face_detector_model', 'yoloface')
    state_manager.init_item('face_detector_size', '640x640')
    state_manager.init_item('face_detector_angles', [0])
    state_manager.init_item('face_detector_score', 0.5)
    state_manager.init_item('face_landmarker_model', '2dfan4')
    state_manager.init_item('face_landmarker_score', 0.5)
    state_manager.init_item('face_recognizer_model', 'arcface_inswapper')

    # Identity and target
    state_manager.init_item('identity_profile_id', source_profile_id)
    state_manager.init_item('target_path', frame_path)  # CRITICAL: pre_check needs this!

    # Load target frame
    print(f"4. Loading target frame...")
    try:
        target_frame = read_image(frame_path)
        print(f"   ✓ Frame loaded: shape={target_frame.shape}")
    except Exception as e:
        print(f"ERROR loading frame: {e}")
        return False

    # Process frame with face swapper
    print(f"5. Processing face swap...")
    try:
        # Pre-process check for preview mode
        if not face_swapper.pre_process('preview'):
            print("ERROR: Face swapper pre-check failed")
            return False

        # Process the frame
        swapped_frame = face_swapper.process_frame({
            'reference_faces': None,
            'source_face': source_face,
            'source_audio_frame': numpy.zeros((1024, 2), dtype=numpy.float32),
            'source_vision_frame': target_frame.copy(),
            'target_vision_frame': target_frame
        })

        print(f"   ✓ Face swap completed: output shape={swapped_frame.shape}")

    except Exception as e:
        print(f"ERROR during face swap: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Save output
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"preview_{lora_model}_frame{random_frame.split('_')[1].split('.')[0]}_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)

    print(f"6. Saving result...")
    try:
        write_image(output_path, swapped_frame)
        print(f"   ✓ Saved to: {output_path}")
    except Exception as e:
        print(f"ERROR saving image: {e}")
        return False

    print(f"\n{'='*60}")
    print(f"✓ PREVIEW TEST PASSED")
    print(f"{'='*60}\n")
    print(f"Test Summary:")
    print(f"  Model: {lora_model}")
    print(f"  Source: {source_profile_id} ({profile.name})")
    print(f"  Frame: {random_frame}")
    print(f"  Output: {output_path}")
    print()

    return True


if __name__ == '__main__':
    success = create_preview_test()
    sys.exit(0 if success else 1)
