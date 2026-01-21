import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_depth_histogram(depth_path):
    print(f"Analyzing {depth_path}...")
    # Read as grayscale
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    
    if depth_map is None:
        print("Error: Could not read file.")
        return

    # Calculate histogram
    # We expect two peaks: one for the face (background/midground) and one for the occlusion (foreground)
    hist = cv2.calcHist([depth_map], [0], None, [256], [0, 256])
    
    # Find peaks (simple approach)
    # Ignore 0 (background) usually
    search_range = hist[10:] 
    max_val = np.max(search_range)
    
    print("\n--- Depth Value Analysis (0-255) ---")
    print(f"Min Value: {np.min(depth_map)}")
    print(f"Max Value: {np.max(depth_map)}")
    print(f"Mean Value: {np.mean(depth_map):.2f}")
    
    # Percentiles to understand distribution
    p25, p50, p75, p90, p95 = np.percentile(depth_map, [25, 50, 75, 90, 95])
    print(f"25th Percentile: {p25}")
    print(f"50th Percentile (Median): {p50}")
    print(f"75th Percentile: {p75}")
    print(f"90th Percentile: {p90}")
    print(f"95th Percentile: {p95}")
    
    print("\nInterpretation:")
    print("The 'Face' is likely around the Median/75th percentile.")
    print("The 'Occlusion' (Corn Dog) should be the brightest values (90th-95th percentile).")
    print(f"Suggested Threshold (Start of Occlusion): {(p75 + p95) / 2:.0f}")

if __name__ == "__main__":
    analyze_depth_histogram("previewtest/frame_depth.png")
