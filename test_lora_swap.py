import cv2
import subprocess
import os
import numpy as np

def run_lora_swap():
    source = "/Users/kendrick/Documents/FS Source/Sam_Generated.png"
    target = "/Users/kendrick/Documents/FS Source/zBam_noOcclusion.png"
    output_path = "previewtest/lora_swap_final.png"
    temp_output = "previewtest/temp_lora_swap.png"
    
    # Run swapper with LoRA
    cmd = [
        "./venv/bin/python", "watserface.py", "headless-run",
        "-s", source,
        "-t", target,
        "-o", temp_output,
        "--processors", "face_swapper", "face_enhancer",
        "--face-swapper-model", "sam_to_zbam_lora",
        "--face-landmarker-model", "2dfan4",
        "--face-mask-types", "region",
        "--face-mask-blur", "0.1",
        "--face-enhancer-model", "restoreformer_plus_plus",
        "--face-enhancer-blend", "80",
        "--output-image-quality", "100"
    ]
    
    print(f"Running LoRA Swap with sam_to_zbam_lora...")
    subprocess.run(cmd, check=True)

    # Post-process: Strict Blend to preserve original pixels outside the mask
    print(f"Applying strict blend to preserve background...")
    swap = cv2.imread(temp_output)
    orig = cv2.imread(target)
    
    if swap is None or orig is None:
        print("Error loading images.")
        return

    # Difference mask
    diff = cv2.absdiff(swap, orig)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=3)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    
    alpha = mask.astype(float) / 255.0
    alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
    
    final = (swap.astype(float) * alpha) + (orig.astype(float) * (1.0 - alpha))
    cv2.imwrite(output_path, final.astype(np.uint8))
    print(f"Result saved to {output_path}")

if __name__ == "__main__":
    run_lora_swap()
