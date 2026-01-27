import zlib
import os

model_path = ".assets/models/trained/sam_to_zbam_lora.onnx"
if not os.path.exists(model_path):
    print("Error: Model file not found.")
    exit(1)

with open(model_path, 'rb') as f:
    model_content = f.read()
model_hash = format(zlib.crc32(model_content), '08x')

hash_path = ".assets/models/trained/sam_to_zbam_lora.hash"
with open(hash_path, 'w') as f:
    f.write(model_hash)

print(f"Updated hash for {model_path}: {model_hash}")
