import cv2
import sys
import os
import numpy as np

# Add the current directory to path
sys.path.append(os.getcwd())

from watserface.depth.estimator import estimate_depth

def generate_depth_map(input_path, output_path):
    print(f"Loading {input_path}...")
    frame = cv2.imread(input_path)
    if frame is None:
        print(f"Error: Could not load {input_path}")
        return

    print("Estimating depth...")
    # estimate_depth returns a float32 array (0..1)
    depth_map = estimate_depth(frame, model_type='midas_small')
    
    # Normalize to 0-255 for visualization/saving
    depth_display = (depth_map * 255).astype(np.uint8)
    
    cv2.imwrite(output_path, depth_display)
    print(f"Saved depth map to {output_path}")

if __name__ == "__main__":
    generate_depth_map(
        "/Users/kendrick/Documents/FS Source/zBam.png",
        "previewtest/frame_depth.png"
    )
