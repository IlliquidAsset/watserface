import json
import os

path = "models/identities/identity_1/profile.json"

try:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Read valid JSON from {path}")
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        
    print(f"Wrote valid JSON back to {path}")
    
except Exception as e:
    print(f"Error: {e}")
