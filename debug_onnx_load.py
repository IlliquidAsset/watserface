import onnxruntime
import os

model_path = ".assets/models/trained/sam_to_zbam_lora.onnx"
print(f"Loading {model_path}...")
print(f"Absolute path: {os.path.abspath(model_path)}")
print(f"Data file exists: {os.path.exists(model_path + '.data')}")

try:
    sess = onnxruntime.InferenceSession(os.path.abspath(model_path), providers=['CPUExecutionProvider'])
    print("Success! Model loaded.")
except Exception as e:
    print(f"Error: {e}")
