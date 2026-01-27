#!/usr/bin/env python3
"""Video swap quality test with iteration until success."""

import os
import sys
import cv2
import numpy as np
import argparse
from pathlib import Path

DEFAULT_SOURCE = "/Users/kendrick/Documents/FS Source/Sam_Generated.png"
DEFAULT_TARGET = "/Users/kendrick/Documents/FS Source/zBambola.mp4"
DEFAULT_OUTPUT_DIR = "/Users/kendrick/Documents/dev/watserface/test_quality"


def init_state():
    from watserface import state_manager
    state_manager.init_item('download_providers', ['github', 'huggingface'])
    state_manager.init_item('execution_device_id', '0')
    state_manager.init_item('execution_providers', ['CPUExecutionProvider'])
    state_manager.init_item('execution_thread_count', 4)
    state_manager.init_item('execution_queue_count', 1)
    state_manager.init_item('video_memory_strategy', 'moderate')
    state_manager.init_item('face_detector_model', 'yolo_face')
    state_manager.init_item('face_detector_size', '640x640')
    state_manager.init_item('face_detector_score', 0.5)
    state_manager.init_item('face_detector_angles', [0])
    state_manager.init_item('face_landmarker_model', '2dfan4')
    state_manager.init_item('face_landmarker_score', 0.5)
    state_manager.init_item('face_swapper_model', 'inswapper_128')
    state_manager.init_item('face_swapper_pixel_boost', '128x128')
    state_manager.init_item('face_swapper_warp_mode', 'affine')
    state_manager.init_item('face_mask_types', ['box'])
    state_manager.init_item('face_mask_blur', 0.3)
    state_manager.init_item('face_mask_padding', [0, 0, 0, 0])
    state_manager.init_item('face_mask_regions', ['skin'])
    state_manager.init_item('face_mask_areas', [])
    state_manager.init_item('processors', ['face_swapper'])
    state_manager.init_item('face_selector_mode', 'one')
    state_manager.init_item('reference_face_distance', 0.6)
    state_manager.init_item('app_context', 'cli')


def compute_frame_diff(frame1, frame2):
    """Compute difference between two frames."""
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
    diff = cv2.absdiff(frame1, frame2)
    return np.mean(diff)


def assess_frame_quality(source_face, swapped_frame, original_frame):
    """Quick quality assessment for a frame.
    
    NOTE: diff_mean is misleading because face region is only ~14% of full frame.
    A diff_mean of 0.7 actually corresponds to ~5 diff in the face region, which is GOOD.
    Primary quality metric should be identity_score (cosine similarity to source).
    """
    from watserface.face_analyser import get_many_faces
    
    diff_mean = compute_frame_diff(swapped_frame, original_frame)
    
    swapped_faces = get_many_faces([swapped_frame])
    original_faces = get_many_faces([original_frame])
    
    if not swapped_faces:
        return {'swap_detected': False, 'diff_mean': diff_mean, 'identity_score': 0, 'identity_shift': 0}
    
    swapped_face = swapped_faces[0]
    original_face = original_faces[0] if original_faces else None
    
    identity_score = 0
    identity_shift = 0
    
    if source_face.embedding is not None and swapped_face.embedding is not None:
        cosine_sim = np.dot(source_face.embedding, swapped_face.embedding) / (
            np.linalg.norm(source_face.embedding) * np.linalg.norm(swapped_face.embedding)
        )
        identity_score = float(cosine_sim)
        
        if original_face is not None and original_face.embedding is not None:
            original_to_source_similarity = np.dot(source_face.embedding, original_face.embedding) / (
                np.linalg.norm(source_face.embedding) * np.linalg.norm(original_face.embedding)
            )
            identity_shift = identity_score - original_to_source_similarity
    
    good_identity_transfer = identity_score > 0.5
    significant_identity_shift = identity_shift > 0.3
    swap_detected = good_identity_transfer or significant_identity_shift
    
    return {
        'swap_detected': swap_detected,
        'diff_mean': float(diff_mean),
        'identity_score': identity_score,
        'identity_shift': identity_shift
    }


def process_video(source_path, target_path, output_path, max_frames=None):
    """Process video with face swap."""
    from watserface.face_analyser import get_many_faces
    from watserface.processors.modules.face_swapper import swap_face
    from watserface.vision import read_static_image
    
    print(f"\nProcessing video: {target_path}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    source_frame = read_static_image(source_path)
    source_faces = get_many_faces([source_frame])
    if not source_faces:
        print("ERROR: No face found in source image")
        return None
    source_face = source_faces[0]
    print(f"Source face detected: embedding shape {source_face.embedding.shape}")
    
    cap = cv2.VideoCapture(target_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {target_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {width}x{height} @ {fps:.2f} fps, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    quality_samples = []
    frame_idx = 0
    swapped_count = 0
    failed_count = 0
    
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        original_frame = frame.copy()
        
        target_faces = get_many_faces([frame])
        
        if target_faces:
            target_face = target_faces[0]
            swapped_frame = swap_face(source_face, target_face, frame)
            
            if frame_idx % 10 == 0:
                quality = assess_frame_quality(source_face, swapped_frame, original_frame)
                quality_samples.append(quality)
                if quality['swap_detected']:
                    swapped_count += 1
                else:
                    failed_count += 1
                print(f"  Frame {frame_idx}/{total_frames}: identity={quality['identity_score']:.3f}, shift={quality.get('identity_shift', 0):.3f}, swap={'YES' if quality['swap_detected'] else 'NO'}")
        else:
            swapped_frame = frame
            failed_count += 1
            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{total_frames}: NO FACE DETECTED")
        
        out.write(swapped_frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    avg_diff = np.mean([q['diff_mean'] for q in quality_samples]) if quality_samples else 0
    avg_identity = np.mean([q['identity_score'] for q in quality_samples]) if quality_samples else 0
    swap_rate = swapped_count / (swapped_count + failed_count) if (swapped_count + failed_count) > 0 else 0
    
    return {
        'total_frames': frame_idx,
        'swapped_frames': swapped_count,
        'failed_frames': failed_count,
        'swap_rate': swap_rate,
        'avg_diff': avg_diff,
        'avg_identity': avg_identity,
        'output_path': output_path
    }


def run_iteration(source_path, target_path, output_dir, iteration, params):
    """Run one iteration with given parameters."""
    from watserface import state_manager
    
    output_name = f"video_swap_iter{iteration:02d}.mp4"
    output_path = os.path.join(output_dir, output_name)
    
    print(f"\n{'='*60}")
    print(f"ITERATION {iteration}")
    print(f"{'='*60}")
    print(f"Parameters: {params}")
    
    state_manager.set_item('face_mask_blur', params.get('mask_blur', 0.3))
    state_manager.set_item('face_swapper_pixel_boost', params.get('pixel_boost', '128x128'))
    
    result = process_video(source_path, target_path, output_path, max_frames=params.get('max_frames'))
    
    if result:
        print(f"\nResults:")
        print(f"  Swap rate: {result['swap_rate']*100:.1f}%")
        print(f"  Avg diff: {result['avg_diff']:.1f}")
        print(f"  Avg identity: {result['avg_identity']:.3f}")
        print(f"  Output: {output_path}")
        
        success = result['swap_rate'] > 0.8 and result['avg_identity'] > 0.5
        result['success'] = success
        result['iteration'] = iteration
        result['params'] = params
        
        return result
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Video swap quality test with iteration')
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Source image path')
    parser.add_argument('--target', default=DEFAULT_TARGET, help='Target video path')
    parser.add_argument('--output', default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--max-iterations', type=int, default=10, help='Maximum iterations')
    parser.add_argument('--max-frames', type=int, default=50, help='Max frames to process per iteration')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*60)
    print("VIDEO SWAP ITERATION TEST")
    print("="*60)
    
    init_state()
    
    param_variations = [
        {'mask_blur': 0.3, 'pixel_boost': '128x128', 'max_frames': args.max_frames},
        {'mask_blur': 0.2, 'pixel_boost': '256x256', 'max_frames': args.max_frames},
        {'mask_blur': 0.4, 'pixel_boost': '256x256', 'max_frames': args.max_frames},
        {'mask_blur': 0.3, 'pixel_boost': '512x512', 'max_frames': args.max_frames},
        {'mask_blur': 0.1, 'pixel_boost': '512x512', 'max_frames': args.max_frames},
        {'mask_blur': 0.5, 'pixel_boost': '256x256', 'max_frames': args.max_frames},
        {'mask_blur': 0.2, 'pixel_boost': '128x128', 'max_frames': args.max_frames},
        {'mask_blur': 0.3, 'pixel_boost': '384x384', 'max_frames': args.max_frames},
        {'mask_blur': 0.15, 'pixel_boost': '384x384', 'max_frames': args.max_frames},
        {'mask_blur': 0.25, 'pixel_boost': '512x512', 'max_frames': args.max_frames},
    ]
    
    results = []
    best_result = None
    
    for i in range(min(args.max_iterations, len(param_variations))):
        params = param_variations[i]
        result = run_iteration(args.source, args.target, args.output, i + 1, params)
        
        if result:
            results.append(result)
            
            if result['success']:
                print(f"\n*** SUCCESS at iteration {i+1}! ***")
                best_result = result
                break
            
            if best_result is None or result['swap_rate'] > best_result['swap_rate']:
                best_result = result
    
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    if best_result:
        print(f"Best result: Iteration {best_result['iteration']}")
        print(f"  Swap rate: {best_result['swap_rate']*100:.1f}%")
        print(f"  Avg identity: {best_result['avg_identity']:.3f}")
        print(f"  Output: {best_result['output_path']}")
        print(f"  Success: {'YES' if best_result.get('success') else 'NO'}")
    else:
        print("No successful iterations")
    
    print(f"\nAll outputs saved to: {args.output}")


if __name__ == '__main__':
    main()
