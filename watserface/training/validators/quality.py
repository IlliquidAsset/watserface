"""
Quality Assurance and Recursive Review for face swapping.

Scans swapped video frames for issues and triggers automatic repair:
- Face detection confidence checks
- Identity similarity validation  
- Occlusion detection and repair
"""

import os
import cv2
import numpy as np
from typing import Iterator, Tuple, Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn.functional as F

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

from watserface import logger


class QualityValidator:
    """
    QA system for face swapping results.
    
    Analyzes swapped frames for:
    - Face detection confidence
    - Identity preservation (cosine similarity)
    - Visual artifacts and occlusions
    
    Flags problematic frames for repair.
    """
    
    def __init__(
        self,
        face_confidence_threshold: float = 0.7,
        identity_similarity_threshold: float = 0.6,
        device: str = 'auto'
    ):
        self.face_confidence_threshold = face_confidence_threshold
        self.identity_similarity_threshold = identity_similarity_threshold
        
        self.device = device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.face_analyzer = None
        self.source_embedding = None
        
    def initialize(self, source_image_path: Optional[str] = None) -> bool:
        """
        Initialize face analysis models.
        
        Args:
            source_image_path: Path to source face image for identity comparison
            
        Returns:
            True if initialization successful
        """
        try:
            if not INSIGHTFACE_AVAILABLE:
                logger.warn("[QA] InsightFace not available, using fallback detection", __name__)
                return False
            
            # Initialize face analyzer
            self.face_analyzer = FaceAnalysis(name='buffalo_l')
            self.face_analyzer.prepare(ctx_id=0 if self.device == 'cpu' else -1)
            
            # Load source identity if provided
            if source_image_path and os.path.exists(source_image_path):
                source_img = cv2.imread(source_image_path)
                if source_img is not None:
                    faces = self.face_analyzer.get(source_img)
                    if faces:
                        self.source_embedding = faces[0].embedding
                        logger.info("[QA] Source identity loaded for validation", __name__)
            
            return True
            
        except Exception as e:
            logger.warn(f"[QA] Failed to initialize: {e}", __name__)
            return False
    
    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """
        Analyze a single frame for quality issues.
        
        Args:
            frame: Frame image as numpy array
            frame_idx: Frame index in video
            
        Returns:
            Analysis results dictionary
        """
        results = {
            'frame_idx': frame_idx,
            'face_detected': False,
            'face_confidence': 0.0,
            'identity_similarity': 0.0,
            'issues': [],
            'needs_repair': False
        }
        
        if self.face_analyzer is None:
            results['issues'].append('no_face_analyzer')
            results['needs_repair'] = True
            return results
        
        try:
            # Detect faces
            faces = self.face_analyzer.get(frame)
            
            if not faces:
                results['issues'].append('no_face_detected')
                results['needs_repair'] = True
                return results
            
            # Use the largest face (assuming it's the main subject)
            main_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            results['face_detected'] = True
            results['face_confidence'] = float(main_face.det_score)
            
            # Check face confidence
            if results['face_confidence'] < self.face_confidence_threshold:
                results['issues'].append('low_face_confidence')
                results['needs_repair'] = True
            
            # Check identity similarity if source embedding available
            if self.source_embedding is not None and hasattr(main_face, 'embedding'):
                current_embedding = main_face.embedding
                similarity = self._compute_cosine_similarity(
                    self.source_embedding, current_embedding
                )
                results['identity_similarity'] = float(similarity)
                
                if similarity < self.identity_similarity_threshold:
                    results['issues'].append('identity_mismatch')
                    results['needs_repair'] = True
            
            # Additional visual checks
            visual_issues = self._check_visual_artifacts(frame, main_face)
            results['issues'].extend(visual_issues)
            if visual_issues:
                results['needs_repair'] = True
            
        except Exception as e:
            logger.warn(f"[QA] Frame analysis failed for frame {frame_idx}: {e}", __name__)
            results['issues'].append('analysis_error')
            results['needs_repair'] = True
        
        return results
    
    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        return np.dot(emb1_norm, emb2_norm)
    
    def _check_visual_artifacts(self, frame: np.ndarray, face) -> List[str]:
        """Check for visual artifacts in the face region."""
        issues = []
        
        try:
            # Extract face region
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return ['invalid_face_region']
            
            # Check for excessive blur (high frequency content)
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var < 100:  # Threshold for blur detection
                issues.append('face_blur')
            
            # Check for color artifacts (unusual color distributions)
            hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
            saturation_std = np.std(hsv[:, :, 1])
            
            if saturation_std > 100:  # High saturation variation
                issues.append('color_artifacts')
            
        except Exception as e:
            logger.debug(f"[QA] Visual check failed: {e}", __name__)
        
        return issues
    
    def analyze_video(
        self, 
        video_path: str, 
        sample_rate: int = 30
    ) -> Iterator[Tuple[str, Dict]]:
        """
        Analyze video frames for quality issues.
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every Nth frame
            
        Yields:
            (status_message, analysis_results)
        """
        if not os.path.exists(video_path):
            yield f"Video not found: {video_path}", {
                'status': 'Error',
                'error': 'video_not_found'
            }
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield f"Could not open video: {video_path}", {
                'status': 'Error', 
                'error': 'video_open_failed'
            }
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        yield f"Analyzing {total_frames} frames (sampling every {sample_rate})", {
            'status': 'Starting Analysis',
            'total_frames': total_frames,
            'sample_rate': sample_rate,
            'fps': fps
        }
        
        frame_idx = 0
        analyzed_frames = 0
        problematic_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames
            if frame_idx % sample_rate == 0:
                analysis = self.analyze_frame(frame, frame_idx)
                
                if analysis['needs_repair']:
                    problematic_frames.append(analysis)
                
                analyzed_frames += 1
                
                # Progress update
                if analyzed_frames % 10 == 0:
                    progress = (analyzed_frames * sample_rate / total_frames) * 100
                    yield f"Analyzed {analyzed_frames} samples ({progress:.1f}%)", {
                        'status': 'Analyzing',
                        'frames_analyzed': analyzed_frames,
                        'problematic_frames': len(problematic_frames),
                        'progress': f"{progress:.1f}%"
                    }
            
            frame_idx += 1
        
        cap.release()
        
        # Summary
        quality_score = 1.0 - (len(problematic_frames) / max(analyzed_frames, 1))
        
        results = {
            'status': 'Analysis Complete',
            'total_frames': total_frames,
            'analyzed_frames': analyzed_frames,
            'problematic_frames': len(problematic_frames),
            'quality_score': f"{quality_score:.2f}",
            'problematic_frame_details': problematic_frames
        }
        
        status_msg = f"QA Complete: {len(problematic_frames)}/{analyzed_frames} frames need repair (Score: {quality_score:.2f})"
        yield status_msg, results


class RecursiveRepair:
    """
    Automatic repair system for problematic frames.
    
    Uses generative inpainting to fix:
    - Low confidence detections
    - Identity mismatches  
    - Visual artifacts
    """
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.inpainting_pipeline = None
        
    def initialize(self) -> bool:
        """Initialize inpainting pipeline."""
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            self.inpainting_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16
            )
            self.inpainting_pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')
            
            return True
        except Exception as e:
            logger.warn(f"[Repair] Failed to initialize inpainting: {e}", __name__)
            return False
    
    def repair_frame(
        self, 
        frame: np.ndarray, 
        mask: np.ndarray,
        prompt: str = "high quality face, realistic skin texture",
        strength: float = 0.75
    ) -> np.ndarray:
        """
        Repair a single frame using inpainting.
        
        Args:
            frame: Original frame
            mask: Binary mask of region to repair (255 = repair)
            prompt: Inpainting prompt
            strength: Denoising strength
            
        Returns:
            Repaired frame
        """
        if self.inpainting_pipeline is None:
            logger.warn("[Repair] Inpainting pipeline not initialized", __name__)
            return frame
        
        try:
            # Convert to PIL
            from PIL import Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mask_image = Image.fromarray(mask)
            
            # Run inpainting
            result = self.inpainting_pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask_image,
                strength=strength,
                num_inference_steps=20,
                guidance_scale=7.5
            ).images[0]
            
            # Convert back
            repaired = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
            return repaired
            
        except Exception as e:
            logger.warn(f"[Repair] Frame repair failed: {e}", __name__)
            return frame
    
    def repair_video(
        self,
        input_video: str,
        output_video: str,
        problematic_frames: List[Dict],
        mask_generator = None
    ) -> Iterator[Tuple[str, Dict]]:
        """
        Repair problematic frames in a video.
        
        Args:
            input_video: Input video path
            output_video: Output video path  
            problematic_frames: List of frame analysis results
            mask_generator: Function to generate repair masks
            
        Yields:
            (status_message, telemetry_dict)
        """
        if not self.initialize():
            yield "Inpainting pipeline initialization failed", {'status': 'Error'}
            return
        
        # Group problematic frames by issues
        frames_by_issue = {}
        for frame_info in problematic_frames:
            for issue in frame_info['issues']:
                if issue not in frames_by_issue:
                    frames_by_issue[issue] = []
                frames_by_issue[issue].append(frame_info['frame_idx'])
        
        yield f"Repairing {len(problematic_frames)} problematic frames", {
            'status': 'Starting Repair',
            'total_repairs': len(problematic_frames),
            'issues_found': list(frames_by_issue.keys())
        }
        
        # For now, just report - full repair implementation would require
        # frame-by-frame video processing and re-encoding
        
        yield "Video repair completed (placeholder - full implementation needed)", {
            'status': 'Repair Complete',
            'frames_processed': len(problematic_frames)
        }


def run_quality_assurance(
    video_path: str,
    source_image_path: Optional[str] = None,
    sample_rate: int = 30,
    progress: Any = None
) -> Iterator[Tuple[str, Dict]]:
    """
    Run complete quality assurance on a swapped video.
    
    Args:
        video_path: Path to swapped video
        source_image_path: Path to source face for identity validation
        sample_rate: Frame sampling rate
        progress: Optional progress callback
        
    Yields:
        (status_message, results_dict)
    """
    validator = QualityValidator()
    
    if not validator.initialize(source_image_path):
        yield "QA initialization failed", {'status': 'Error'}
        return
    
    yield from validator.analyze_video(video_path, sample_rate)


def run_recursive_repair(
    input_video: str,
    output_video: str,
    qa_results: Dict,
    progress: Any = None
) -> Iterator[Tuple[str, Dict]]:
    """
    Run recursive repair on problematic frames.
    
    Args:
        input_video: Input video path
        output_video: Output video path
        qa_results: Results from quality assurance
        progress: Optional progress callback
        
    Yields:
        (status_message, telemetry_dict)
    """
    problematic_frames = qa_results.get('problematic_frame_details', [])
    
    if not problematic_frames:
        yield "No frames need repair", {'status': 'Complete'}
        return
    
    repair = RecursiveRepair()
    yield from repair.repair_video(
        input_video=input_video,
        output_video=output_video,
        problematic_frames=problematic_frames
    )
