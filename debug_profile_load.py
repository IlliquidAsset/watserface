from watserface.identity_profile import IdentityProfileManager, get_identity_manager
import sys
import os

print(f"CWD: {os.getcwd()}")
print(f"Checking models/identities/identity_1/profile.json")
try:
    with open("models/identities/identity_1/profile.json", "r") as f:
        print(f"File content start: {f.read(50)}")
except Exception as e:
    print(f"File read error: {e}")

try:
    manager = get_identity_manager()
    print("Manager initialized")
    profile = manager.source_intelligence.load_profile("identity_1")
    if profile:
        print("✅ Profile loaded successfully")
        print(f"Mean embedding shape: {len(profile.embedding_mean)}")
    else:
        print("❌ Profile load returned None")
except Exception as e:
    print(f"❌ Exception loading profile: {e}")
    import traceback
    traceback.print_exc()
