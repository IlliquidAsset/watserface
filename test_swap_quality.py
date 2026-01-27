#!/usr/bin/env python3
"""
Automated face swap quality assessment and iteration script.

Uses DeepFace for CG/deepfake detection confidence scoring.
Iterates parameter tweaks until quality thresholds met or max iterations reached.

Usage:
    python3 test_swap_quality.py --output test_quality --iterations 10
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

DEFAULT_SOURCE = "/Users/kendrick/Documents/FS Source/Sam_Generated.png"
DEFAULT_TARGET = "/Users/kendrick/Documents/FS Source/zBam_noOcclusion.png"
DEFAULT_OCCLUDED = "/Users/kendrick/Documents/FS Source/zBam.png"

SOURCE_VIDEO = "/Users/kendrick/Documents/FS Source/VID_20200509_150723.mp4"
TARGET_VIDEO = "/Users/kendrick/Documents/FS Source/zBambola.mp4"


def init_state(warp_mode='tps'):
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
    state_manager.init_item('face_swapper_pixel_boost', '512x512')
    state_manager.init_item('face_swapper_warp_mode', warp_mode)
    state_manager.init_item('face_mask_types', ['box'])
    state_manager.init_item('face_mask_blur', 0.3)
    state_manager.init_item('face_mask_padding', [0, 0, 0, 0])
    state_manager.init_item('face_mask_regions', ['skin'])
    state_manager.init_item('face_mask_areas', [])
    state_manager.init_item('processors', ['face_swapper'])
    state_manager.init_item('face_selector_mode', 'one')
    state_manager.init_item('reference_face_distance', 0.6)
    state_manager.init_item('app_context', 'cli')


def compute_metrics(original, processed):
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    gray_orig = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    ssim_score = structural_similarity(gray_orig, gray_proc, data_range=gray_proc.max() - gray_proc.min())
    psnr_score = peak_signal_noise_ratio(original, processed, data_range=255)
    
    return {
        'ssim': float(ssim_score),
        'psnr': float(psnr_score)
    }


def get_face_mask_from_landmarks(img_shape, landmarks_68):
    """Generate face mask from 68 landmarks using convex hull"""
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if landmarks_68 is None:
        return None
    
    hull = cv2.convexHull(landmarks_68.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def detect_boundary_artifacts(image, face_mask):
    """Detect visible seams at mask boundary using gradient analysis"""
    if face_mask is None:
        return 0.0, 0.0
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_eroded = cv2.erode(face_mask, kernel, iterations=2)
    mask_dilated = cv2.dilate(face_mask, kernel, iterations=2)
    boundary = mask_dilated - mask_eroded
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    boundary_gradients = gradient_mag[boundary > 0]
    if len(boundary_gradients) == 0:
        return 0.0, 0.0
    
    mean_gradient = np.mean(boundary_gradients)
    max_gradient = np.max(boundary_gradients)
    
    return float(mean_gradient), float(max_gradient)


def detect_color_mismatch(image, face_mask):
    """Compare LAB color statistics between face and background"""
    if face_mask is None:
        return 0.0
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    face_pixels = lab[face_mask > 128]
    bg_pixels = lab[face_mask <= 128]
    
    if len(face_pixels) == 0 or len(bg_pixels) == 0:
        return 0.0
    
    face_mean = np.mean(face_pixels, axis=0)
    bg_mean = np.mean(bg_pixels, axis=0)
    
    color_distance = np.sqrt(np.sum((face_mean - bg_mean) ** 2))
    return float(color_distance)


def detect_blur_mismatch(image, face_mask):
    """Compare sharpness between face region and background"""
    if face_mask is None:
        return 0.0, 0.0, 1.0
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    face_blur = np.var(laplacian[face_mask > 128])
    bg_blur = np.var(laplacian[face_mask <= 128])
    
    if face_blur < 1:
        face_blur = 1
    
    blur_ratio = bg_blur / face_blur
    return float(face_blur), float(bg_blur), float(blur_ratio)


def detect_frequency_artifacts(image):
    """Detect artifacts using FFT high-frequency analysis"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    high_pass = gray - blurred
    
    fft = np.fft.fft2(high_pass)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = magnitude.shape
    ch, cw = h // 2, w // 2
    
    low_band = magnitude[ch-30:ch+30, cw-30:cw+30]
    mid_band = magnitude[ch-80:ch+80, cw-80:cw+80]
    high_band = magnitude.copy()
    high_band[ch-80:ch+80, cw-80:cw+80] = 0
    
    low_energy = np.mean(low_band) if low_band.size > 0 else 1
    mid_energy = np.mean(mid_band) if mid_band.size > 0 else 0
    high_energy = np.mean(high_band) if high_band.size > 0 else 0
    
    freq_ratio = (mid_energy + high_energy) / (low_energy + 1e-6)
    return float(freq_ratio), float(high_energy)


def detect_color_quantization(image):
    """Detect dithering/limited color palette (comic book effect)"""
    unique_colors = len(np.unique(image.reshape(-1, 3), axis=0))
    total_pixels = image.shape[0] * image.shape[1]
    color_ratio = unique_colors / total_pixels
    
    b, g, r = cv2.split(image)
    unique_b = len(np.unique(b))
    unique_g = len(np.unique(g))
    unique_r = len(np.unique(r))
    avg_unique_per_channel = (unique_b + unique_g + unique_r) / 3
    
    is_quantized = avg_unique_per_channel < 50 or color_ratio < 0.01
    return unique_colors, avg_unique_per_channel, color_ratio, is_quantized


def detect_pattern_repetition(image, original_image=None, face_mask=None):
    """Detect duplication artifacts by analyzing the difference image"""
    
    if original_image is not None:
        orig_resized = cv2.resize(original_image, (image.shape[1], image.shape[0]))
        diff = cv2.absdiff(image, orig_resized)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        diff_mean = np.mean(diff_gray)
        diff_std = np.std(diff_gray)
        diff_max = np.max(diff_gray)
        
        _, thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        num_changed_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        change_ratio = num_changed_pixels / total_pixels
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours = [c for c in contours if cv2.contourArea(c) > 1000]
        
        is_severe = diff_mean > 15 or change_ratio > 0.25 or len(large_contours) > 15
        
        return len(large_contours), float(diff_mean), is_severe
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = (gray - gray.mean()) / (gray.std() + 1e-6)
        fft = np.fft.fft2(gray)
        power_spectrum = np.abs(fft) ** 2
        autocorr = np.fft.ifft2(power_spectrum).real
        autocorr = np.fft.fftshift(autocorr)
        h, w = autocorr.shape
        ch, cw = h // 2, w // 2
        center_value = autocorr[ch, cw]
        autocorr[ch-20:ch+20, cw-20:cw+20] = 0
        threshold = center_value * 0.3
        num_peaks = np.sum(autocorr > threshold)
        
        return int(num_peaks), float(threshold), num_peaks > 50


def detect_multiple_faces(image):
    """Detect if multiple face-like regions exist (double rendering)"""
    from watserface.face_analyser import get_many_faces
    faces = get_many_faces([image])
    return len(faces), faces


def detect_catastrophic_failure(image):
    """Check for NaN, Inf, or extreme values indicating corrupted output"""
    if not np.isfinite(image).all():
        return True, "NaN/Inf values detected"
    
    if image.min() == image.max():
        return True, "Uniform image (no variation)"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    if laplacian.var() < 1:
        return True, "Extremely blurry (Laplacian var < 1)"
    
    return False, None


def assess_cg_confidence(source_path, processed_img, output_dir, target_face=None, original_target=None):
    """
    Comprehensive artifact detection for face swap quality assessment.
    Returns artifact scores where HIGHER = MORE ARTIFACTS = WORSE QUALITY.
    """
    from watserface.face_analyser import get_many_faces
    from watserface.vision import read_static_image
    
    is_catastrophic, catastrophic_reason = detect_catastrophic_failure(processed_img)
    if is_catastrophic:
        return {
            'cg_confidence': 100.0,
            'identity_verified': False,
            'identity_score': 0.0,
            'face_detected': False,
            'catastrophic_failure': True,
            'catastrophic_reason': catastrophic_reason,
            'analysis': {}
        }
    
    unique_colors, avg_unique_channel, color_ratio, is_quantized = detect_color_quantization(processed_img)
    num_repetitions, repetition_strength, has_repetition = detect_pattern_repetition(processed_img, original_target)
    num_faces, faces_list = detect_multiple_faces(processed_img)
    
    source_frame = read_static_image(source_path)
    source_faces = get_many_faces([source_frame])
    source_face = source_faces[0] if source_faces else None
    processed_face = faces_list[0] if faces_list else None
    
    face_mask = None
    if processed_face is not None:
        landmarks_68 = processed_face.landmark_set.get('68')
        if landmarks_68 is not None:
            face_mask = get_face_mask_from_landmarks(processed_img.shape, landmarks_68)
    
    boundary_mean, boundary_max = detect_boundary_artifacts(processed_img, face_mask)
    color_mismatch = detect_color_mismatch(processed_img, face_mask)
    face_blur, bg_blur, blur_ratio = detect_blur_mismatch(processed_img, face_mask)
    freq_ratio, high_freq = detect_frequency_artifacts(processed_img)
    
    if source_face is not None and processed_face is not None:
        source_emb = source_face.embedding
        proc_emb = processed_face.embedding
        if source_emb is not None and proc_emb is not None:
            cosine_sim = np.dot(source_emb, proc_emb) / (np.linalg.norm(source_emb) * np.linalg.norm(proc_emb))
            identity_score = float(cosine_sim)
            identity_verified = cosine_sim > 0.4
            face_detected = True
        else:
            identity_score = 0.0
            identity_verified = False
            face_detected = True
    else:
        identity_score = 0.0
        identity_verified = False
        face_detected = processed_face is not None
    
    quantization_penalty = 100 if is_quantized else min(100, max(0, 50 - avg_unique_channel) * 2)
    repetition_penalty = 100 if has_repetition else 0
    multiface_penalty = min(100, (num_faces - 1) * 50) if num_faces > 1 else 0
    
    boundary_penalty = min(100, boundary_mean * 2)
    color_penalty = min(100, color_mismatch * 2)
    blur_penalty = min(100, abs(np.log10(blur_ratio + 0.01)) * 30) if blur_ratio > 0 else 50
    freq_penalty = min(100, freq_ratio * 10)
    identity_penalty = (1 - identity_score) * 100
    
    has_major_artifact = is_quantized or has_repetition or num_faces > 1
    
    if has_major_artifact:
        artifact_score = max(
            quantization_penalty,
            repetition_penalty,
            multiface_penalty,
            70
        )
    else:
        artifact_score = (
            boundary_penalty * 0.20 +
            color_penalty * 0.15 +
            blur_penalty * 0.10 +
            freq_penalty * 0.10 +
            identity_penalty * 0.25 +
            quantization_penalty * 0.10 +
            repetition_penalty * 0.05 +
            multiface_penalty * 0.05
        )
    
    return {
        'cg_confidence': float(min(100, max(0, artifact_score))),
        'identity_verified': identity_verified,
        'identity_score': identity_score,
        'face_detected': face_detected,
        'catastrophic_failure': False,
        'has_major_artifact': has_major_artifact,
        'analysis': {
            'unique_colors': int(unique_colors),
            'avg_unique_channel': float(avg_unique_channel),
            'is_quantized': is_quantized,
            'quantization_penalty': float(quantization_penalty),
            'num_repetitions': int(num_repetitions),
            'repetition_strength': float(repetition_strength),
            'has_repetition': has_repetition,
            'repetition_penalty': float(repetition_penalty),
            'num_faces_detected': int(num_faces),
            'multiface_penalty': float(multiface_penalty),
            'boundary_mean': float(boundary_mean),
            'boundary_max': float(boundary_max),
            'boundary_penalty': float(boundary_penalty),
            'color_mismatch': float(color_mismatch),
            'color_penalty': float(color_penalty),
            'face_blur': float(face_blur),
            'bg_blur': float(bg_blur),
            'blur_ratio': float(blur_ratio),
            'blur_penalty': float(blur_penalty),
            'freq_ratio': float(freq_ratio),
            'high_freq_energy': float(high_freq),
            'freq_penalty': float(freq_penalty),
            'identity_score': float(identity_score),
            'identity_penalty': float(identity_penalty)
        }
    }


def perform_swap(source_face, target_face, target_frame, warp_mode, crop_size=(512, 512), 
                 model_template='ffhq_512', landmark_count=18, scale_factor=0.7):
    from watserface import state_manager
    from watserface.face_helper import thin_plate_spline_warp, warp_face_by_face_landmark_5
    
    state_manager.set_item('face_swapper_warp_mode', warp_mode)
    
    crop_frame, affine_matrix = warp_face_by_face_landmark_5(
        target_frame, 
        target_face.landmark_set.get('5/68'),
        model_template,
        crop_size
    )
    
    if warp_mode == 'tps' and target_face.landmark_set.get('68') is not None:
        source_landmark_68 = source_face.landmark_set.get('68')
        target_landmark_68 = target_face.landmark_set.get('68')
        
        if source_landmark_68 is not None and target_landmark_68 is not None:
            target_68_crop = cv2.transform(
                target_landmark_68.reshape(1, -1, 2), 
                affine_matrix
            ).reshape(-1, 2)
            
            key_indices = [0, 8, 16, 17, 21, 22, 26, 36, 39, 42, 45, 27, 30, 33, 48, 51, 54, 57]
            key_indices = key_indices[:min(landmark_count, len(key_indices))]
            
            source_68_norm = source_landmark_68.copy()
            source_bbox = np.array([
                source_68_norm[:, 0].min(), source_68_norm[:, 1].min(),
                source_68_norm[:, 0].max(), source_68_norm[:, 1].max()
            ])
            source_center = np.array([
                (source_bbox[0] + source_bbox[2]) / 2, 
                (source_bbox[1] + source_bbox[3]) / 2
            ])
            source_scale = max(source_bbox[2] - source_bbox[0], source_bbox[3] - source_bbox[1])
            
            crop_h, crop_w = crop_frame.shape[:2]
            source_68_normalized = (source_68_norm - source_center) / source_scale * min(crop_w, crop_h) * scale_factor + np.array([crop_w / 2, crop_h / 2])
            
            source_key = source_68_normalized[key_indices].astype(np.float32)
            target_key = target_68_crop[key_indices].astype(np.float32)
            
            crop_frame = thin_plate_spline_warp(crop_frame, source_key, target_key, (crop_w, crop_h))
    
    return crop_frame, affine_matrix


def run_full_swap(source_path, target_path, output_dir, warp_mode='tps'):
    from watserface.face_analyser import get_many_faces, get_one_face
    from watserface.processors.modules.face_swapper import swap_face
    from watserface.vision import read_static_image
    
    # Clear image cache to ensure fresh reads
    if hasattr(read_static_image, 'cache_clear'):
        read_static_image.cache_clear()
    
    source_frame = read_static_image(source_path)
    target_frame = read_static_image(target_path)
    
    if source_frame is None:
        print(f"ERROR: Cannot read source: {source_path}")
        return None
    if target_frame is None:
        print(f"ERROR: Cannot read target: {target_path}")
        return None
    
    print(f"Source shape: {source_frame.shape}")
    print(f"Target shape: {target_frame.shape}")
    
    source_faces = get_many_faces([source_frame])
    target_faces = get_many_faces([target_frame])
    
    if not source_faces:
        print("ERROR: No face detected in source")
        return None
    if not target_faces:
        print("ERROR: No face detected in target")
        return None
    
    source_face = source_faces[0]
    target_face = target_faces[0]
    
    print(f"Source landmarks: 5/68={source_face.landmark_set.get('5/68') is not None}, 68={source_face.landmark_set.get('68') is not None}, 478={source_face.landmark_set.get('478') is not None}")
    print(f"Target landmarks: 5/68={target_face.landmark_set.get('5/68') is not None}, 68={target_face.landmark_set.get('68') is not None}, 478={target_face.landmark_set.get('478') is not None}")
    
    from watserface import state_manager
    state_manager.set_item('face_swapper_warp_mode', warp_mode)
    
    result_frame = swap_face(source_face, target_face, target_frame.copy())
    
    return result_frame, source_face, target_face, target_frame


def iterate_optimization(source_path, target_path, output_dir, max_iterations=10, 
                        threshold_ssim=0.85, threshold_cg=60):
    from watserface.face_analyser import get_many_faces
    from watserface.vision import read_static_image
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("PHASE 1: Baseline Comparison (Affine vs TPS)")
    print("=" * 60)
    
    init_state('affine')
    result_affine = run_full_swap(source_path, target_path, output_dir, 'affine')
    
    if result_affine is None:
        print("ERROR: Affine swap failed")
        return
    
    affine_frame, source_face, target_face, target_frame = result_affine
    cv2.imwrite(os.path.join(output_dir, 'baseline_affine.jpg'), affine_frame)
    print(f"Saved baseline_affine.jpg")
    
    init_state('tps')
    result_tps = run_full_swap(source_path, target_path, output_dir, 'tps')
    
    if result_tps is None:
        print("ERROR: TPS swap failed")
        return
    
    tps_frame = result_tps[0]
    cv2.imwrite(os.path.join(output_dir, 'baseline_tps.jpg'), tps_frame)
    print(f"Saved baseline_tps.jpg")
    
    h, w = target_frame.shape[:2]
    aspect = w / h
    comp_height = 720
    comp_width = int(comp_height * aspect)
    comparison = np.hstack([
        cv2.resize(target_frame, (comp_width, comp_height)),
        cv2.resize(affine_frame, (comp_width, comp_height)),
        cv2.resize(tps_frame, (comp_width, comp_height))
    ])
    cv2.imwrite(os.path.join(output_dir, 'comparison_original_affine_tps.jpg'), comparison)
    print(f"Saved comparison_original_affine_tps.jpg (Original | Affine | TPS) - {comp_width}x{comp_height} each")
    
    print("\n" + "=" * 60)
    print("PHASE 2: Quality Assessment")
    print("=" * 60)
    
    affine_metrics = compute_metrics(target_frame, affine_frame)
    tps_metrics = compute_metrics(target_frame, tps_frame)
    
    print(f"\nAffine metrics: SSIM={affine_metrics['ssim']:.4f}, PSNR={affine_metrics['psnr']:.2f}")
    print(f"TPS metrics:    SSIM={tps_metrics['ssim']:.4f}, PSNR={tps_metrics['psnr']:.2f}")
    
    def print_artifact_analysis(name, result):
        print(f"\n{'='*70}")
        print(f"{name} ARTIFACT ANALYSIS")
        print(f"{'='*70}")
        
        if result.get('catastrophic_failure'):
            print(f"*** CATASTROPHIC FAILURE: {result.get('catastrophic_reason')} ***")
            print(f"ARTIFACT SCORE: 100% (UNUSABLE)")
            return
        
        score = result['cg_confidence']
        grade = "GOOD" if score < 30 else "OK" if score < 50 else "POOR" if score < 70 else "FAIL"
        print(f"ARTIFACT SCORE: {score:.1f}% [{grade}] (target: <40%)")
        
        if result.get('has_major_artifact'):
            print(f"*** MAJOR ARTIFACTS DETECTED ***")
        
        a = result.get('analysis', {})
        if a:
            print(f"\n-- CRITICAL CHECKS (any fail = bad output) --")
            quant_status = "FAIL" if a.get('is_quantized') else "PASS"
            rep_status = "FAIL" if a.get('has_repetition') else "PASS"
            face_status = "FAIL" if a.get('num_faces_detected', 1) > 1 else "PASS"
            print(f"  Color Quantization (dithering):  [{quant_status}]  unique_colors={a.get('unique_colors', 0):,}, per_channel={a.get('avg_unique_channel', 0):.0f}")
            print(f"  Diff Artifacts (vs original):    [{rep_status}]  regions={a.get('num_repetitions', 0)}, diff_mean={a.get('repetition_strength', 0):.1f}")
            print(f"  Multiple Faces (double render):  [{face_status}]  faces_found={a.get('num_faces_detected', 0)}")
            
            print(f"\n-- QUALITY METRICS --")
            print(f"  Identity Match:    {a.get('identity_score', 0):.3f} ({'PASS' if result.get('identity_verified') else 'FAIL'}, need >0.4)")
            print(f"  Boundary Seams:    {a.get('boundary_penalty', 0):5.1f}%  (gradient={a.get('boundary_mean', 0):.1f})")
            print(f"  Color Mismatch:    {a.get('color_penalty', 0):5.1f}%  (LAB dist={a.get('color_mismatch', 0):.1f})")
            print(f"  Blur Mismatch:     {a.get('blur_penalty', 0):5.1f}%  (ratio={a.get('blur_ratio', 0):.2f})")
            print(f"  Frequency Noise:   {a.get('freq_penalty', 0):5.1f}%  (ratio={a.get('freq_ratio', 0):.2f})")
    
    print("\nRunning artifact detection...")
    cg_result = assess_cg_confidence(source_path, tps_frame, output_dir, original_target=target_frame)
    print_artifact_analysis("TPS", cg_result)
    
    cg_result_affine = assess_cg_confidence(source_path, affine_frame, output_dir, original_target=target_frame)
    print_artifact_analysis("AFFINE", cg_result_affine)
    
    print("\n" + "=" * 60)
    print("PHASE 3: Iteration Optimization")
    print("=" * 60)
    
    best_frame = tps_frame
    best_metrics = {**tps_metrics, **cg_result}
    best_score = cg_result['cg_confidence'] + (1 - tps_metrics['ssim']) * 100
    best_iteration = 0
    best_params = {'landmark_count': 18, 'scale_factor': 0.7}
    
    stagnant_count = 0
    
    for iteration in range(1, max_iterations + 1):
        print(f"\nIteration {iteration}/{max_iterations}")
        
        landmark_count = min(18 + iteration * 2, 18)
        scale_factor = 0.7 + iteration * 0.03
        
        print(f"Params: landmarks={landmark_count}, scale={scale_factor:.2f}")
        
        crop, affine_matrix = perform_swap(
            source_face, target_face, target_frame, 
            'tps', 
            landmark_count=landmark_count,
            scale_factor=scale_factor
        )
        
        metrics = compute_metrics(target_frame, crop)
        cg = assess_cg_confidence(source_path, crop, output_dir, original_target=target_frame)
        
        combined_score = cg['cg_confidence'] + (1 - metrics['ssim']) * 100
        
        print(f"Results: SSIM={metrics['ssim']:.4f}, CG={cg['cg_confidence']:.2f}%, Score={combined_score:.2f}")
        
        if metrics['ssim'] > threshold_ssim and cg['cg_confidence'] < threshold_cg:
            print(f"Thresholds met at iteration {iteration}")
            best_frame = crop
            best_metrics = {**metrics, **cg}
            best_iteration = iteration
            best_params = {'landmark_count': landmark_count, 'scale_factor': scale_factor}
            break
        
        if combined_score < best_score - 0.5:
            best_score = combined_score
            best_frame = crop
            best_metrics = {**metrics, **cg}
            best_iteration = iteration
            best_params = {'landmark_count': landmark_count, 'scale_factor': scale_factor}
            stagnant_count = 0
        else:
            stagnant_count += 1
        
        if stagnant_count >= 3:
            print(f"No improvement for 3 iterations, stopping early")
            break
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    cv2.imwrite(os.path.join(output_dir, f'best_result_iter{best_iteration}.jpg'), best_frame)
    
    report = {
        'source': source_path,
        'target': target_path,
        'best_iteration': int(best_iteration),
        'best_params': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in best_params.items()},
        'metrics': {
            'ssim': float(best_metrics.get('ssim', 0)),
            'psnr': float(best_metrics.get('psnr', 0)),
            'cg_confidence': float(best_metrics.get('cg_confidence', 0)),
            'identity_verified': bool(best_metrics.get('identity_verified', False)),
            'face_detected': bool(best_metrics.get('face_detected', False))
        },
        'thresholds': {
            'ssim': float(threshold_ssim),
            'cg': float(threshold_cg)
        },
        'baseline_comparison': {
            'affine_ssim': float(affine_metrics['ssim']),
            'tps_ssim': float(tps_metrics['ssim'])
        }
    }
    
    with open(os.path.join(output_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nBest iteration: {best_iteration}")
    print(f"Best params: {best_params}")
    print(f"Final SSIM: {best_metrics.get('ssim', 'N/A'):.4f}")
    print(f"Final CG Confidence: {best_metrics.get('cg_confidence', 'N/A'):.2f}%")
    print(f"\nOutputs saved to {output_dir}/")
    print(f"  - baseline_affine.jpg")
    print(f"  - baseline_tps.jpg")
    print(f"  - comparison_original_affine_tps.jpg")
    print(f"  - best_result_iter{best_iteration}.jpg")
    print(f"  - report.json")


def main():
    parser = argparse.ArgumentParser(description='Automated face swap quality assessment')
    parser.add_argument('--source', default=DEFAULT_SOURCE, help='Source image path')
    parser.add_argument('--target', default=DEFAULT_TARGET, help='Target image path')
    parser.add_argument('--output', default='test_quality', help='Output directory')
    parser.add_argument('--iterations', type=int, default=10, help='Max iterations')
    parser.add_argument('--threshold-ssim', type=float, default=0.85, help='SSIM threshold')
    parser.add_argument('--threshold-cg', type=float, default=60, help='CG confidence threshold')
    args = parser.parse_args()
    
    print("=" * 60)
    print("FACE SWAP QUALITY ASSESSMENT & OPTIMIZATION")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Output: {args.output}")
    print(f"Max iterations: {args.iterations}")
    print(f"Thresholds: SSIM>{args.threshold_ssim}, CG<{args.threshold_cg}%")
    
    iterate_optimization(
        args.source, 
        args.target, 
        args.output, 
        args.iterations,
        args.threshold_ssim,
        args.threshold_cg
    )


if __name__ == '__main__':
    main()
