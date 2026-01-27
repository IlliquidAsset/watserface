#!/usr/bin/env python3
"""
Test script for TPS warp alignment improvements.

Tests the new TPS (Thin Plate Spline) warping mode that uses 68 landmarks
instead of 5-point affine transform for better eye/cheekbone alignment.

Usage:
    python test_tps_alignment.py --source /path/to/source.jpg --target /path/to/target.jpg
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from watserface import state_manager
from watserface.face_analyser import get_many_faces, get_one_face
from watserface.face_helper import (
    convert_to_face_landmark_5,
    convert_to_face_landmark_5_from_478,
    thin_plate_spline_warp,
    warp_face_by_face_landmark_5,
)
from watserface.vision import read_static_image


def init_state():
    """Initialize minimal state for face detection."""
    state_manager.init_item('face_detector_model', 'yolo_face')
    state_manager.init_item('face_detector_score', 0.5)
    state_manager.init_item('face_detector_angles', [0])
    state_manager.init_item('face_landmarker_model', 'mediapipe')
    state_manager.init_item('face_landmarker_score', 0.5)


def draw_landmarks(image, landmarks, color=(0, 255, 0), radius=2):
    """Draw landmarks on image."""
    img = image.copy()
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        cv2.circle(img, (x, y), radius, color, -1)
    return img


def test_landmark_extraction(source_path, target_path, output_dir):
    """Test landmark extraction from MediaPipe 478."""
    print("\n=== Testing Landmark Extraction ===")
    
    source_frame = read_static_image(source_path)
    target_frame = read_static_image(target_path)
    
    if source_frame is None:
        print(f"ERROR: Could not read source image: {source_path}")
        return False
    if target_frame is None:
        print(f"ERROR: Could not read target image: {target_path}")
        return False
    
    print(f"Source: {source_frame.shape}")
    print(f"Target: {target_frame.shape}")
    
    source_faces = get_many_faces([source_frame])
    target_faces = get_many_faces([target_frame])
    
    if not source_faces:
        print("ERROR: No face detected in source")
        return False
    if not target_faces:
        print("ERROR: No face detected in target")
        return False
    
    source_face = source_faces[0]
    target_face = target_faces[0]
    
    print(f"\nSource face landmarks:")
    print(f"  - 5 points: {source_face.landmark_set.get('5') is not None}")
    print(f"  - 5/68 points: {source_face.landmark_set.get('5/68') is not None}")
    print(f"  - 68 points: {source_face.landmark_set.get('68') is not None}")
    print(f"  - 478 points: {source_face.landmark_set.get('478') is not None}")
    
    if source_face.landmark_set.get('478') is not None:
        landmarks_478 = source_face.landmark_set.get('478')
        print(f"  - 478 shape: {landmarks_478.shape}")
        
        landmark_5 = convert_to_face_landmark_5_from_478(landmarks_478)
        print(f"  - Converted 5 from 478: {landmark_5.shape}")
        print(f"  - Eye centers (iris): [{landmark_5[0]}, {landmark_5[1]}]")
    
    source_vis = draw_landmarks(source_frame, source_face.landmark_set.get('5/68'), (0, 255, 0), 3)
    if source_face.landmark_set.get('68') is not None:
        source_vis = draw_landmarks(source_vis, source_face.landmark_set.get('68'), (255, 0, 0), 2)
    
    target_vis = draw_landmarks(target_frame, target_face.landmark_set.get('5/68'), (0, 255, 0), 3)
    if target_face.landmark_set.get('68') is not None:
        target_vis = draw_landmarks(target_vis, target_face.landmark_set.get('68'), (255, 0, 0), 2)
    
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, 'source_landmarks.jpg'), source_vis)
    cv2.imwrite(os.path.join(output_dir, 'target_landmarks.jpg'), target_vis)
    print(f"\nSaved landmark visualizations to {output_dir}")
    
    return True


def test_warp_comparison(source_path, target_path, output_dir):
    """Compare affine vs TPS warping results."""
    print("\n=== Testing Warp Comparison (Affine vs TPS) ===")
    
    source_frame = read_static_image(source_path)
    target_frame = read_static_image(target_path)
    
    source_faces = get_many_faces([source_frame])
    target_faces = get_many_faces([target_frame])
    
    if not source_faces or not target_faces:
        print("ERROR: Could not detect faces")
        return False
    
    source_face = source_faces[0]
    target_face = target_faces[0]
    
    crop_size = (512, 512)
    model_template = 'ffhq_512'
    
    print("\n1. Affine warp (5-point)...")
    crop_affine, affine_matrix = warp_face_by_face_landmark_5(
        target_frame, 
        target_face.landmark_set.get('5/68'),
        model_template,
        crop_size
    )
    
    has_68 = (source_face.landmark_set.get('68') is not None and 
              target_face.landmark_set.get('68') is not None)
    
    if has_68:
        print("\n2. TPS warp (68-point)...")
        target_landmark_68 = target_face.landmark_set.get('68')
        source_landmark_68 = source_face.landmark_set.get('68')
        
        target_68_crop = cv2.transform(
            target_landmark_68.reshape(1, -1, 2), 
            affine_matrix
        ).reshape(-1, 2)
        
        key_indices = [0, 8, 16, 17, 21, 22, 26, 36, 39, 42, 45, 27, 30, 33, 48, 51, 54, 57]
        
        source_68_norm = source_landmark_68.copy()
        source_bbox = np.array([
            source_68_norm[:, 0].min(), source_68_norm[:, 1].min(),
            source_68_norm[:, 0].max(), source_68_norm[:, 1].max()
        ])
        source_center = np.array([(source_bbox[0] + source_bbox[2]) / 2, (source_bbox[1] + source_bbox[3]) / 2])
        source_scale = max(source_bbox[2] - source_bbox[0], source_bbox[3] - source_bbox[1])
        
        crop_h, crop_w = crop_affine.shape[:2]
        source_68_normalized = (source_68_norm - source_center) / source_scale * min(crop_w, crop_h) * 0.7 + np.array([crop_w / 2, crop_h / 2])
        
        source_key = source_68_normalized[key_indices].astype(np.float32)
        target_key = target_68_crop[key_indices].astype(np.float32)
        
        crop_tps = thin_plate_spline_warp(crop_affine, source_key, target_key, (crop_w, crop_h))
        
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'crop_affine.jpg'), crop_affine)
        cv2.imwrite(os.path.join(output_dir, 'crop_tps.jpg'), crop_tps)
        
        comparison = np.hstack([crop_affine, crop_tps])
        cv2.imwrite(os.path.join(output_dir, 'comparison_affine_vs_tps.jpg'), comparison)
        print(f"\nSaved comparison to {output_dir}/comparison_affine_vs_tps.jpg")
        print("Left: Affine (5-point), Right: TPS (68-point)")
    else:
        print("\nWARNING: 68 landmarks not available, skipping TPS comparison")
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, 'crop_affine.jpg'), crop_affine)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test TPS alignment improvements')
    parser.add_argument('--source', '-s', required=True, help='Source face image')
    parser.add_argument('--target', '-t', required=True, help='Target face image')
    parser.add_argument('--output', '-o', default='previewtest/tps_test', help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("TPS Alignment Test")
    print("=" * 60)
    
    init_state()
    
    success = True
    success = test_landmark_extraction(args.source, args.target, args.output) and success
    success = test_warp_comparison(args.source, args.target, args.output) and success
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
