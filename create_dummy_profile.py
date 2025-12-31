#!/usr/bin/env python3
"""Quick script to create a dummy profile for Samantha"""

import json
import os
from pathlib import Path
from datetime import datetime

# Create profile directory
profile_dir = Path("models/identities/samantha")
profile_dir.mkdir(parents=True, exist_ok=True)

# Create dummy profile with zero embedding (will be replaced later)
profile = {
    "id": "samantha",
    "name": "Samantha",
    "embedding_mean": [0.0] * 512,  # Dummy embedding
    "quality_stats": {
        "total_processed": 2000,
        "final_embedding_count": 100,
        "source_count": 1
    },
    "created_at": datetime.now().isoformat(),
    "last_used": datetime.now().isoformat()
}

# Save profile
profile_file = profile_dir / "profile.json"
with open(profile_file, 'w') as f:
    json.dump(profile, f, indent=2)

print(f"✅ Created dummy profile at: {profile_file}")
print(f"   Note: This profile has a zero embedding vector and should be replaced")
print(f"         by retraining the Samantha identity model.")
