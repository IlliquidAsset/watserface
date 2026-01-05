"""
Identity Builder - Face Set Management with auto-extraction and temporal smoothing.

This module handles:
1. Face extraction from video with intelligent frame sampling
2. Temporal smoothing using Savitzky-Golay filter for jitter reduction
3. Face set persistence to disk for reuse
4. Quality-based face selection (blur, pose, occlusion detection)
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid

import cv2
import numpy

from watserface import logger, state_manager
from watserface.filesystem import is_video, is_image, resolve_relative_path
from watserface.face_analyser import get_one_face, get_many_faces
from watserface.face_helper import convert_to_face_landmark_5
from watserface.types import Face, FaceLandmark68, VisionFrame


# Directory for persisted face sets
FACE_SET_DIR = resolve_relative_path('../.assets/face_sets')


@dataclass
class ExtractedFace:
    """A single extracted face with metadata."""
    frame_index: int
    face_index: int
    image: numpy.ndarray
    landmarks_68: FaceLandmark68
    quality_score: float
    blur_score: float
    pose_score: float
    embedding: Optional[numpy.ndarray] = None


@dataclass
class FaceSet:
    """A collection of extracted faces from a source identity."""
    id: str
    name: str
    faces: List[ExtractedFace] = field(default_factory=list)
    source_paths: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    # Smoothed landmarks for video sources
    smoothed_landmarks: Optional[numpy.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'source_paths': self.source_paths,
            'created_at': self.created_at,
            'face_count': len(self.faces)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceSet':
        return cls(
            id=data['id'],
            name=data['name'],
            source_paths=data.get('source_paths', []),
            created_at=data.get('created_at', time.time())
        )


class IdentityBuilder:
    """
    Builds and manages identity face sets from source media.
    
    Workflow:
    1. Add source media (images/videos)
    2. Extract faces with quality filtering
    3. Apply temporal smoothing for video sources
    4. Persist face set for training
    """
    
    def __init__(self, face_set_dir: Optional[str] = None):
        self.face_set_dir = Path(face_set_dir or FACE_SET_DIR)
        self.face_set_dir.mkdir(parents=True, exist_ok=True)
        
        # Extraction parameters
        self.min_face_size = 64  # Minimum face dimension
        self.min_quality_score = 0.3
        self.frames_per_second = 2  # For video sampling
        self.max_faces_per_video = 100
        
        # Smoothing parameters (Savitzky-Golay)
        self.smoothing_window = 7  # Must be odd
        self.smoothing_polyorder = 2
    
    def extract_from_sources(
        self,
        name: str,
        source_paths: List[str],
        progress_callback: Optional[callable] = None
    ) -> Generator[Tuple[str, Dict], None, FaceSet]:
        """
        Extract faces from multiple source files.
        
        Args:
            name: Identity name
            source_paths: List of image/video paths
            progress_callback: Optional callback for progress updates
            
        Yields:
            (status_message, telemetry_dict)
            
        Returns:
            FaceSet with extracted faces
        """
        face_set = FaceSet(
            id=str(uuid.uuid4()),
            name=name,
            source_paths=source_paths
        )
        
        total_sources = len(source_paths)
        
        for idx, source_path in enumerate(source_paths):
            if not os.path.exists(source_path):
                yield f'Source not found: {source_path}', {'error': True}
                continue
            
            progress = (idx + 1) / total_sources
            yield f'Processing {os.path.basename(source_path)} ({idx + 1}/{total_sources})', {
                'progress': progress,
                'current_source': source_path
            }
            
            if is_video(source_path):
                for msg, data in self._extract_from_video(source_path, face_set):
                    yield msg, data
            elif is_image(source_path):
                for msg, data in self._extract_from_image(source_path, face_set):
                    yield msg, data
            else:
                yield f'Unsupported format: {source_path}', {'error': True}
        
        # Apply temporal smoothing if we have video-sourced faces
        if face_set.faces and any(is_video(p) for p in source_paths):
            yield 'Applying temporal smoothing...', {}
            self._apply_temporal_smoothing(face_set)
        
        yield f'Extraction complete: {len(face_set.faces)} faces', {
            'face_count': len(face_set.faces),
            'face_set_id': face_set.id
        }
        
        return face_set
    
    def _extract_from_video(
        self,
        video_path: str,
        face_set: FaceSet
    ) -> Generator[Tuple[str, Dict], None, None]:
        """Extract faces from video with frame sampling."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield f'Failed to open video: {video_path}', {'error': True}
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / self.frames_per_second))
        
        yield f'Video: {total_frames} frames at {fps:.1f} FPS, sampling every {frame_interval} frames', {
            'total_frames': total_frames,
            'fps': fps,
            'frame_interval': frame_interval
        }
        
        extracted_count = 0
        frame_index = 0
        
        while cap.isOpened() and extracted_count < self.max_faces_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_index % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face = get_one_face([frame_rgb])
                
                if face and self._is_quality_acceptable(face, frame_rgb):
                    extracted_face = self._create_extracted_face(
                        frame_index, 0, frame_rgb, face
                    )
                    face_set.faces.append(extracted_face)
                    extracted_count += 1
                    
                    if extracted_count % 10 == 0:
                        yield f'Extracted {extracted_count} faces...', {
                            'extracted_count': extracted_count,
                            'frame_index': frame_index
                        }
            
            frame_index += 1
        
        cap.release()
        yield f'Video extraction complete: {extracted_count} faces', {
            'extracted_count': extracted_count
        }
    
    def _extract_from_image(
        self,
        image_path: str,
        face_set: FaceSet
    ) -> Generator[Tuple[str, Dict], None, None]:
        """Extract faces from a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            yield f'Failed to read image: {image_path}', {'error': True}
            return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = get_many_faces([frame_rgb])
        
        if not faces:
            yield f'No faces found in: {os.path.basename(image_path)}', {'face_count': 0}
            return
        
        for i, face in enumerate(faces):
            if self._is_quality_acceptable(face, frame_rgb):
                extracted_face = self._create_extracted_face(
                    0, i, frame_rgb, face
                )
                face_set.faces.append(extracted_face)
        
        yield f'Extracted {len(faces)} face(s) from image', {'face_count': len(faces)}
    
    def _create_extracted_face(
        self,
        frame_index: int,
        face_index: int,
        frame: VisionFrame,
        face: Face
    ) -> ExtractedFace:
        """Create an ExtractedFace from detection results."""
        # Crop face region with padding
        bbox = face.bounding_box.astype(int)
        h, w = frame.shape[:2]
        
        # Add 20% padding
        pad_x = int((bbox[2] - bbox[0]) * 0.2)
        pad_y = int((bbox[3] - bbox[1]) * 0.2)
        
        x1 = max(0, bbox[0] - pad_x)
        y1 = max(0, bbox[1] - pad_y)
        x2 = min(w, bbox[2] + pad_x)
        y2 = min(h, bbox[3] + pad_y)
        
        face_crop = frame[y1:y2, x1:x2].copy()
        
        # Calculate quality scores
        blur_score = self._calculate_blur_score(face_crop)
        pose_score = self._calculate_pose_score(face)
        quality_score = (blur_score + pose_score) / 2
        
        landmarks_68 = face.landmark_set.get('68', face.landmark_set.get('68/5'))
        
        return ExtractedFace(
            frame_index=frame_index,
            face_index=face_index,
            image=face_crop,
            landmarks_68=landmarks_68,
            quality_score=quality_score,
            blur_score=blur_score,
            pose_score=pose_score,
            embedding=face.embedding
        )
    
    def _is_quality_acceptable(self, face: Face, frame: VisionFrame) -> bool:
        """Check if face meets minimum quality requirements."""
        bbox = face.bounding_box
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        
        # Size check
        if face_width < self.min_face_size or face_height < self.min_face_size:
            return False
        
        # Score check
        detector_score = face.score_set.get('detector', 0) if face.score_set else 0
        if detector_score < self.min_quality_score:
            return False
        
        return True
    
    def _calculate_blur_score(self, face_crop: VisionFrame) -> float:
        """Calculate blur score using Laplacian variance (higher = sharper)."""
        if face_crop.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (empirically determined thresholds)
        score = min(1.0, laplacian_var / 500)
        return score
    
    def _calculate_pose_score(self, face: Face) -> float:
        """Calculate pose score (higher = more frontal)."""
        # Use face angle - 0 degrees is frontal
        angle = face.angle if face.angle else 0
        
        # Normalize: 0 degrees = 1.0, 90 degrees = 0.0
        score = 1.0 - min(abs(angle), 90) / 90
        return score
    
    def _apply_temporal_smoothing(self, face_set: FaceSet) -> None:
        """
        Apply Savitzky-Golay temporal smoothing to landmarks.
        
        This reduces jitter in video-extracted faces while preserving
        genuine facial movements.
        """
        if len(face_set.faces) < self.smoothing_window:
            return
        
        try:
            from scipy.signal import savgol_filter
        except ImportError:
            logger.warn('scipy not available, skipping temporal smoothing')
            return
        
        # Stack all landmarks: (N, 68, 2) or (N, 68, 3)
        landmarks_array = numpy.array([
            f.landmarks_68 for f in face_set.faces
            if f.landmarks_68 is not None
        ])
        
        if landmarks_array.shape[0] < self.smoothing_window:
            return
        
        # Apply filter along time axis (axis=0) for each coordinate
        smoothed = savgol_filter(
            landmarks_array,
            window_length=self.smoothing_window,
            polyorder=self.smoothing_polyorder,
            axis=0
        )
        
        for i, face in enumerate(face_set.faces):
            if face.landmarks_68 is not None and i < smoothed.shape[0]:
                face.landmarks_68 = smoothed[i].astype(numpy.float32)
        
        face_set.smoothed_landmarks = smoothed
    
    def save_face_set(self, face_set: FaceSet) -> str:
        """
        Persist face set to disk.
        
        Returns:
            Path to saved face set directory
        """
        set_dir = self.face_set_dir / face_set.id
        set_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = set_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(face_set.to_dict(), f, indent=2)
        
        # Save face images
        faces_dir = set_dir / 'faces'
        faces_dir.mkdir(exist_ok=True)
        
        for i, face in enumerate(face_set.faces):
            face_path = faces_dir / f'face_{i:04d}.png'
            # Convert RGB to BGR for OpenCV
            cv2.imwrite(str(face_path), cv2.cvtColor(face.image, cv2.COLOR_RGB2BGR))
            
            # Save landmarks
            if face.landmarks_68 is not None:
                lm_path = faces_dir / f'landmarks_{i:04d}.npy'
                numpy.save(str(lm_path), face.landmarks_68)
            
            # Save embedding
            if face.embedding is not None:
                emb_path = faces_dir / f'embedding_{i:04d}.npy'
                numpy.save(str(emb_path), face.embedding)
        
        return str(set_dir)
    
    def load_face_set(self, face_set_id: str) -> Optional[FaceSet]:
        """Load a face set from disk."""
        set_dir = self.face_set_dir / face_set_id
        
        if not set_dir.exists():
            return None
        
        metadata_path = set_dir / 'metadata.json'
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            face_set = FaceSet.from_dict(json.load(f))
        
        # Load face images
        faces_dir = set_dir / 'faces'
        if faces_dir.exists():
            face_files = sorted(faces_dir.glob('face_*.png'))
            
            for i, face_path in enumerate(face_files):
                image = cv2.imread(str(face_path))
                if image is None:
                    continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Load landmarks
                lm_path = faces_dir / f'landmarks_{i:04d}.npy'
                landmarks = numpy.load(str(lm_path)) if lm_path.exists() else None
                
                # Load embedding
                emb_path = faces_dir / f'embedding_{i:04d}.npy'
                embedding = numpy.load(str(emb_path)) if emb_path.exists() else None
                
                face_set.faces.append(ExtractedFace(
                    frame_index=i,
                    face_index=0,
                    image=image,
                    landmarks_68=landmarks,
                    quality_score=1.0,
                    blur_score=1.0,
                    pose_score=1.0,
                    embedding=embedding
                ))
        
        return face_set
    
    def list_face_sets(self) -> List[Dict[str, Any]]:
        """List all saved face sets."""
        face_sets = []
        
        for set_dir in self.face_set_dir.iterdir():
            if not set_dir.is_dir():
                continue
            
            metadata_path = set_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    face_sets.append(json.load(f))
        
        return sorted(face_sets, key=lambda x: x.get('created_at', 0), reverse=True)
    
    def delete_face_set(self, face_set_id: str) -> bool:
        """Delete a face set from disk."""
        set_dir = self.face_set_dir / face_set_id
        
        if set_dir.exists():
            shutil.rmtree(str(set_dir))
            return True
        
        return False
    
    def get_best_faces(
        self,
        face_set: FaceSet,
        count: int = 10,
        diversity: bool = True
    ) -> List[ExtractedFace]:
        """
        Select the best faces from a face set.
        
        Args:
            face_set: Source face set
            count: Number of faces to select
            diversity: Whether to ensure temporal diversity
            
        Returns:
            List of best faces
        """
        if not face_set.faces:
            return []
        
        # Sort by quality
        sorted_faces = sorted(
            face_set.faces,
            key=lambda f: f.quality_score,
            reverse=True
        )
        
        if not diversity:
            return sorted_faces[:count]
        
        # Select diverse faces (spread across frames)
        selected = []
        total_frames = max(f.frame_index for f in face_set.faces) + 1
        
        if total_frames <= count:
            return sorted_faces[:count]
        
        # Divide into temporal buckets
        bucket_size = total_frames // count
        
        for i in range(count):
            bucket_start = i * bucket_size
            bucket_end = (i + 1) * bucket_size if i < count - 1 else total_frames
            
            # Find best face in bucket
            bucket_faces = [
                f for f in sorted_faces
                if bucket_start <= f.frame_index < bucket_end and f not in selected
            ]
            
            if bucket_faces:
                selected.append(bucket_faces[0])
            elif sorted_faces:
                # Fallback: take next best unused
                for f in sorted_faces:
                    if f not in selected:
                        selected.append(f)
                        break
        
        return selected


def create_identity_builder() -> IdentityBuilder:
    """Factory function to create IdentityBuilder with default settings."""
    return IdentityBuilder()
