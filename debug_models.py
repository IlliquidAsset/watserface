import sys
import os

# Add watserface to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from watserface import state_manager
from watserface.processors.modules import face_swapper, face_enhancer

# Init state
state_manager.init_item('download_providers', ['huggingface', 'github'])
state_manager.init_item('face_swapper_model', 'inswapper_128_fp16')
state_manager.init_item('face_enhancer_model', 'codeformer')
state_manager.init_item('execution_providers', ['CoreMLExecutionProvider', 'CPUExecutionProvider'])

print("Checking Face Swapper options...")
swapper_options = face_swapper.get_model_options()
if swapper_options:
    print(f"✅ Swapper options found: {swapper_options.get('type')}")
else:
    print("❌ Swapper options is None!")

print("Checking Face Enhancer options...")
enhancer_options = face_enhancer.get_model_options()
if enhancer_options:
    print(f"✅ Enhancer options found: {enhancer_options.get('template')}")
else:
    print("❌ Enhancer options is None!")
