import onnx
import os

model_path = "models/face_sets/faceset_512e84d4_1768337182/frames/sam_to_zbam_lora.onnx"
output_path = ".assets/models/trained/sam_to_zbam_lora.onnx"

print(f"Loading model {model_path}...")
model = onnx.load(model_path)

print("Saving model with internal data...")
onnx.save(model, output_path)

print(f"Done. Saved to {output_path}")
print(f"New size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
