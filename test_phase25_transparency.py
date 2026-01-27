#!/usr/bin/env python3
"""Phase 2.5 Transparency Handling Test

Tests the complete Phase 2.5 pipeline with clean baseline first,
then with occluded targets.
"""
import cv2
import numpy as np
import os
from pathlib import Path

DEPTH_THRESHOLD = 0.74
BLUR_STRENGTH = (5, 5)
TEMPORAL_WINDOW = 5
OUTPUT_DIR = "previewtest"

# Test configurations - clean baseline FIRST, then occluded
TEST_CONFIGS = {
    "clean_baseline": {
        "target": "previewtest/swap_clean_target.png",
        "swap": "previewtest/clean_2dfan4.png",
        "depth": None,  # Will generate
        "description": "Clean baseline (no occlusion)"
    },
    "mayo_occluded": {
        "target": "/Users/kendrick/Documents/FS Source/zBam.png",
        "swap": "previewtest/solution_dirty_swap.png",
        "depth": "previewtest/frame_depth.png",
        "description": "Mayo occlusion test"
    }
}


def run_single_test(config_name: str, config: dict) -> dict:
    """Run a single transparency test configuration."""
    print(f"\n{'='*60}")
    print(f"TEST: {config['description']}")
    print(f"{'='*60}")
    
    results = {"name": config_name, "success": False}
    
    target_path = config["target"]
    swap_path = config["swap"]
    
    if not os.path.exists(target_path):
        print(f"  SKIP: Target not found: {target_path}")
        results["error"] = "Target not found"
        return results
    
    target = cv2.imread(target_path)
    if target is None:
        print(f"  ERROR: Failed to load target")
        results["error"] = "Failed to load target"
        return results
    print(f"  Target: {target.shape}")
    
    if os.path.exists(swap_path):
        dirty_swap = cv2.imread(swap_path)
        print(f"  Swap: {dirty_swap.shape}")
    else:
        dirty_swap = target.copy()
        print(f"  Swap: Using target as fallback")
    
    if dirty_swap.shape[:2] != target.shape[:2]:
        dirty_swap = cv2.resize(dirty_swap, (target.shape[1], target.shape[0]))
        print(f"  Resized swap to match target")
    
    from watserface.depth.dkt_estimator import DKTEstimator
    from watserface.processors.modules.transparency_handler import TransparencyHandler
    from watserface.inpainting.controlnet_optimizer import ControlNetOptimizer
    
    estimator = DKTEstimator()
    handler = TransparencyHandler(depth_threshold=DEPTH_THRESHOLD, blur_strength=BLUR_STRENGTH)
    optimizer = ControlNetOptimizer()
    
    if config["depth"] and os.path.exists(config["depth"]):
        depth_map = cv2.imread(config["depth"], cv2.IMREAD_GRAYSCALE)
        depth_map = depth_map.astype(np.float32) / 255.0
        print(f"  Loaded depth map: {depth_map.shape}")
    else:
        print(f"  Generating depth map...")
        try:
            from watserface.depth.estimator import estimate_depth
            depth_map = estimate_depth(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
            print(f"  Generated depth: {depth_map.shape}")
        except Exception as e:
            h, w = target.shape[:2]
            depth_map = np.ones((h, w), dtype=np.float32) * 0.5
            print(f"  Using uniform depth (MiDaS unavailable)")
    
    alpha = estimator.estimate_alpha(target, depth_map, threshold=DEPTH_THRESHOLD)
    print(f"  Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
    
    result = handler.process_frame(target, dirty_swap, depth_map)
    
    ssim_vs_target = optimizer.compute_ssim(result, target)
    ssim_vs_swap = optimizer.compute_ssim(result, dirty_swap)
    
    print(f"\n  RESULTS:")
    print(f"    SSIM vs Target: {ssim_vs_target:.4f}")
    print(f"    SSIM vs Swap:   {ssim_vs_swap:.4f}")
    
    prefix = f"phase25_{config_name}"
    
    alpha_vis = (alpha * 255).astype(np.uint8)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_alpha.png", alpha_vis)
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_result.png", result)
    
    h, w = target.shape[:2]
    scale = min(600 / w, 400 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    target_s = cv2.resize(target, (new_w, new_h))
    swap_s = cv2.resize(dirty_swap, (new_w, new_h))
    result_s = cv2.resize(result, (new_w, new_h))
    alpha_s = cv2.cvtColor(cv2.resize(alpha_vis, (new_w, new_h)), cv2.COLOR_GRAY2BGR)
    
    row1 = np.hstack([target_s, swap_s])
    row2 = np.hstack([alpha_s, result_s])
    grid = np.vstack([row1, row2])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(grid, "Target (Original)", (10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(grid, "Swap Input", (new_w + 10, 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(grid, "Alpha Mask", (10, new_h + 25), font, 0.6, (0, 255, 0), 2)
    cv2.putText(grid, f"Result (SSIM: {ssim_vs_target:.3f})", (new_w + 10, new_h + 25), font, 0.6, (0, 255, 0), 2)
    
    cv2.imwrite(f"{OUTPUT_DIR}/{prefix}_comparison.png", grid)
    
    print(f"\n  OUTPUTS:")
    print(f"    {OUTPUT_DIR}/{prefix}_alpha.png")
    print(f"    {OUTPUT_DIR}/{prefix}_result.png")
    print(f"    {OUTPUT_DIR}/{prefix}_comparison.png")
    
    results["success"] = True
    results["ssim_target"] = ssim_vs_target
    results["ssim_swap"] = ssim_vs_swap
    
    return results


def run_all_tests():
    """Run all test configurations."""
    print("="*60)
    print("PHASE 2.5 TRANSPARENCY TEST SUITE")
    print("="*60)
    print("\nRunning clean baseline FIRST, then occluded tests")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
    for config_name in ["clean_baseline", "mayo_occluded"]:
        if config_name in TEST_CONFIGS:
            result = run_single_test(config_name, TEST_CONFIGS[config_name])
            all_results.append(result)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for r in all_results:
        status = "PASS" if r.get("success") else "FAIL"
        ssim = r.get("ssim_target", 0)
        print(f"  [{status}] {r['name']}: SSIM={ssim:.4f}" if r.get("success") else f"  [{status}] {r['name']}: {r.get('error', 'Unknown')}")
    
    return all(r.get("success") for r in all_results)


def run_video_test():
    """Test video processing with temporal coherence."""
    print("\n" + "="*60)
    print("VIDEO TEMPORAL COHERENCE TEST")
    print("="*60)
    
    from watserface.processors.modules.transparency_handler import TransparencyHandler
    handler = TransparencyHandler(temporal_window=5)
    
    num_frames = 10
    h, w = 200, 300
    
    frames = []
    dirty_swaps = []
    depth_maps = []
    
    for i in range(num_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 128
        frame[50:150, 100:200] = [200, 100, 50]
        frames.append(frame)
        
        swap = frame.copy()
        swap[50:150, 100:200] = [50, 100, 200]
        dirty_swaps.append(swap)
        
        depth = np.zeros((h, w), dtype=np.float32)
        offset = i * 3
        depth[30 + offset:70 + offset, 80:220] = 0.9
        depth = cv2.GaussianBlur(depth, (11, 11), 0)
        depth_maps.append(depth)
    
    results = handler.process_video(frames, dirty_swaps, depth_maps)
    consistency = handler.compute_temporal_consistency(results)
    
    print(f"  Frames processed: {len(results)}")
    print(f"  Temporal consistency: {consistency:.4f}")
    print(f"  Status: {'PASS' if consistency > 0.9 else 'FAIL'} (target >0.9)")
    
    return consistency > 0.9


if __name__ == "__main__":
    import sys
    
    success1 = run_all_tests()
    success2 = run_video_test()
    
    print("\n" + "="*60)
    if success1 and success2:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)
    
    sys.exit(0 if (success1 and success2) else 1)
