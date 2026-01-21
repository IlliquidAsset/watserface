#!/usr/bin/env python3
"""Quick test to check what's in the identity profile"""

import sys
sys.path.insert(0, '/Users/kendrick/Documents/dev/watserface')

from watserface.identity_profile import get_identity_manager

# Load identity_1 profile
manager = get_identity_manager()
profile = manager.source_intelligence.load_profile('identity_1')

if profile:
    print(f"Profile loaded: {profile.name}")
    print(f"Source files: {profile.source_files}")
    print(f"Embedding mean shape: {len(profile.embedding_mean) if profile.embedding_mean else 'None'}")
    print(f"Embedding std shape: {len(profile.embedding_std) if profile.embedding_std else 'None'}")
    print(f"Is ephemeral: {profile.is_ephemeral}")
    print(f"Face set ID: {profile.face_set_id}")
else:
    print("Profile not found!")
