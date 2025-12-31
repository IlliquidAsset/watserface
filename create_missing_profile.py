#!/usr/bin/env python3
"""
Create Identity Profile from Existing Training Dataset

This script creates an identity profile JSON for models that were trained
before the profile creation feature was added.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add watserface to path
sys.path.insert(0, os.path.dirname(__file__))

from watserface import logger, identity_profile
from watserface.face_analyser import get_one_face
from watserface.vision import read_static_image
from watserface.filesystem import resolve_file_paths


def create_profile_from_frames(dataset_dir: str, model_name: str) -> bool:
    """
    Extract embeddings from training frames and create identity profile.

    Args:
        dataset_dir: Path to training dataset directory (.jobs/training_dataset_identity/)
        model_name: Name of the model (e.g., 'Samantha')

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Creating identity profile for '{model_name}'...", __name__)

    # Get all frame files
    frame_paths = resolve_file_paths(dataset_dir)

    if not frame_paths:
        logger.error(f"No frames found in {dataset_dir}", __name__)
        return False

    logger.info(f"Found {len(frame_paths)} frames (will process max 100)", __name__)

    # Extract embeddings from frames (limit to 100 for performance)
    embeddings = []

    for frame_path in frame_paths[:100]:
        if frame_path.endswith(('.jpg', '.png')):
            try:
                frame = read_static_image(frame_path)
                face = get_one_face([frame])

                if face and face.embedding is not None:
                    embeddings.append(face.embedding)

                    if len(embeddings) % 10 == 0:
                        logger.info(f"Processed {len(embeddings)} frames...", __name__)
            except Exception as e:
                logger.debug(f"Skipping frame {frame_path}: {e}", __name__)
                continue

    if not embeddings:
        logger.error("No embeddings extracted from frames", __name__)
        return False

    logger.info(f"Extracted {len(embeddings)} embeddings from {len(frame_paths)} frames", __name__)

    # Calculate mean embedding
    embedding_mean = np.mean(embeddings, axis=0).tolist()

    # Create identity profile
    manager = identity_profile.get_identity_manager()
    profile = identity_profile.IdentityProfile(
        id=model_name.lower().replace(' ', '_'),
        name=model_name,
        embedding_mean=embedding_mean,
        quality_stats={
            'total_processed': len(frame_paths),
            'final_embedding_count': len(embeddings),
            'source_count': 1  # Unknown, set to 1
        }
    )

    # Save profile
    manager.source_intelligence.save_profile(profile)
    logger.info(f"✅ Saved identity profile '{model_name}' with {len(embeddings)} embeddings", __name__)

    return True


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python3 create_missing_profile.py <model_name> [face_set_id]")
        print("Example: python3 create_missing_profile.py Samantha faceset_migrated_1767130169")
        sys.exit(1)

    model_name = sys.argv[1]

    # Check if Face Set ID provided
    if len(sys.argv) >= 3:
        face_set_id = sys.argv[2]
        from watserface.face_set import get_face_set_manager
        face_set_manager = get_face_set_manager()
        dataset_dir = face_set_manager.get_face_set_frames_path(face_set_id)
    else:
        # Fallback to old .jobs path
        dataset_dir = os.path.join('.jobs', 'training_dataset_identity')

    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}", __name__)
        logger.error(f"Use: python3 create_missing_profile.py {model_name} <face_set_id>", __name__)
        sys.exit(1)

    # Check if profile already exists
    manager = identity_profile.get_identity_manager()
    profile_id = model_name.lower().replace(' ', '_')
    existing_profile = manager.source_intelligence.load_profile(profile_id)

    if existing_profile:
        logger.warn(f"Profile '{model_name}' already exists!", __name__)
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            logger.info("Cancelled.", __name__)
            sys.exit(0)

    # Create profile
    success = create_profile_from_frames(dataset_dir, model_name)

    if success:
        logger.info(f"✅ Profile created successfully!", __name__)
        logger.info(f"   Location: models/identities/{profile_id}/profile.json", __name__)
        logger.info(f"   The profile should now appear in the Modeler tab dropdown.", __name__)
    else:
        logger.error("Failed to create profile", __name__)
        sys.exit(1)


if __name__ == '__main__':
    main()
