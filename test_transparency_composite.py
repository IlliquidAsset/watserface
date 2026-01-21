import cv2
import numpy as np
import os

def run_composite_test():
    # --- Configuration ---
    # Adjust strictly based on the specific depth map intensity of the occlusion
    DEPTH_THRESHOLD_MIN = 0.74  # Calibrated from analyze_depth.py (188/255)
    DEPTH_THRESHOLD_MAX = 1.0  # End of the "liquid"
    BLUR_STRENGTH = (5, 5)     # Softens the mask edges
    
    # --- Paths ---
    target_path = "/Users/kendrick/Documents/FS Source/zBam.png"
    swap_dirty_path = "previewtest/solution_dirty_swap.png"
    depth_path = "previewtest/frame_depth.png"
    output_path = "previewtest/solution_final.png"
    debug_path = "previewtest/debug_mask_solution.png"

    if not os.path.exists(swap_dirty_path):
        print(f"Error: {swap_dirty_path} not found.")
        return
    if not os.path.exists(depth_path):
        print(f"Error: {depth_path} not found.")
        return

    # --- Load Assets ---
    target_original = cv2.imread(target_path).astype(float)
    swap_dirty = cv2.imread(swap_dirty_path).astype(float)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0

    print(f"Loaded assets. Target shape: {target_original.shape}")

    # --- 1. Generate Transmission Mask ---
    # Isolate pixels that are "closer" than the face (the liquid)
    _, liquid_mask = cv2.threshold(depth_map, DEPTH_THRESHOLD_MIN, DEPTH_THRESHOLD_MAX, cv2.THRESH_BINARY)
    
    # Soften the mask to simulate alpha transparency/feathering
    liquid_alpha = cv2.GaussianBlur(liquid_mask, BLUR_STRENGTH, 0)
    
    # Normalize alpha to 0.0 - 1.0 range and expand dimensions
    liquid_alpha = liquid_alpha / 255.0
    liquid_alpha = np.repeat(liquid_alpha[:, :, np.newaxis], 3, axis=2)

    # --- 2. The Composite Math ---
    # Formula: Final = (Swap * (1 - Alpha)) + (Original * Alpha)
    # Explanation: Where Alpha is 1 (liquid), show Original. Where Alpha is 0 (face), show Swap.
    # Where Alpha is 0.5 (semi-transparent), blend them.
    
    final_composite = (swap_dirty * (1.0 - liquid_alpha)) + (target_original * liquid_alpha)

    # Clip and convert to uint8
    final_composite = np.clip(final_composite, 0, 255).astype(np.uint8)

    # --- 3. Save Output ---
    cv2.imwrite(output_path, final_composite)
    cv2.imwrite(debug_path, (liquid_mask * 255).astype(np.uint8))
    
    print("✓ Processing Complete.")
    print(f"  > Output: {output_path}")
    print(f"  > Debug: {debug_path} (Check this to tune DEPTH_THRESHOLD)")

if __name__ == "__main__":
    run_composite_test()
