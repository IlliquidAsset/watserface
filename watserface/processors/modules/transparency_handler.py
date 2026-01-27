"""TransparencyHandler for video processing with temporal coherence.

Implements the "Mayonnaise Strategy" for compositing face swaps under 
semi-transparent occlusions with temporal smoothing for video.
"""
from typing import List, Optional, Tuple, Generator
import numpy
import cv2

from watserface.types import VisionFrame


class TransparencyHandler:
    """Handles transparency compositing for face swaps under occlusions.
    
    Implements the compositing formula:
    Final = (Dirty_Swap * (1 - Alpha)) + (Original_Target * Alpha)
    
    With temporal smoothing for video to reduce flicker.
    """
    
    DEFAULT_DEPTH_THRESHOLD = 0.74
    DEFAULT_BLUR_STRENGTH = (5, 5)
    DEFAULT_TEMPORAL_WINDOW = 5
    
    def __init__(
        self,
        depth_threshold: float = 0.74,
        blur_strength: Tuple[int, int] = (5, 5),
        temporal_window: int = 5
    ):
        """Initialize TransparencyHandler.
        
        Args:
            depth_threshold: Depth threshold for occlusion detection (default 0.74 = 188/255)
            blur_strength: Gaussian blur kernel for mask softening
            temporal_window: Window size for temporal smoothing
        """
        self.depth_threshold = depth_threshold
        self.blur_strength = blur_strength
        self.temporal_window = temporal_window
        
        self._alpha_buffer: List[numpy.ndarray] = []
    
    def process_frame(
        self,
        original: VisionFrame,
        dirty_swap: VisionFrame,
        depth_map: numpy.ndarray
    ) -> VisionFrame:
        """Process single frame with transparency compositing.
        
        Args:
            original: Original target frame (with occlusion)
            dirty_swap: Face-swapped frame (swapped "through" occlusion)
            depth_map: Depth map for the frame
        
        Returns:
            Composited frame with occlusion preserved
        """
        alpha = self._compute_alpha(depth_map)
        return self._composite(original, dirty_swap, alpha)
    
    def process_video(
        self,
        frames: List[VisionFrame],
        dirty_swaps: List[VisionFrame],
        depth_maps: List[numpy.ndarray],
        chunk_size: int = 100
    ) -> List[VisionFrame]:
        """Process video with temporal coherence.
        
        Args:
            frames: Original target frames
            dirty_swaps: Face-swapped frames
            depth_maps: Depth maps for each frame
            chunk_size: Frames per chunk for memory efficiency
        
        Returns:
            List of composited frames
        """
        if not (len(frames) == len(dirty_swaps) == len(depth_maps)):
            raise ValueError("frames, dirty_swaps, and depth_maps must have same length")
        
        if len(frames) == 0:
            return []
        
        results = []
        self._alpha_buffer.clear()
        
        for chunk_start in range(0, len(frames), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(frames))
            
            chunk_results = self._process_chunk(
                frames[chunk_start:chunk_end],
                dirty_swaps[chunk_start:chunk_end],
                depth_maps[chunk_start:chunk_end]
            )
            results.extend(chunk_results)
        
        return results
    
    def process_video_generator(
        self,
        frames: List[VisionFrame],
        dirty_swaps: List[VisionFrame],
        depth_maps: List[numpy.ndarray]
    ) -> Generator[VisionFrame, None, None]:
        """Process video as generator for memory efficiency.
        
        Yields composited frames one at a time.
        """
        if not (len(frames) == len(dirty_swaps) == len(depth_maps)):
            raise ValueError("frames, dirty_swaps, and depth_maps must have same length")
        
        self._alpha_buffer.clear()
        
        alpha_maps = [self._compute_alpha(dm) for dm in depth_maps]
        smoothed_alphas = self._apply_temporal_smoothing(alpha_maps)
        
        for i in range(len(frames)):
            yield self._composite(frames[i], dirty_swaps[i], smoothed_alphas[i])
    
    def _process_chunk(
        self,
        frames: List[VisionFrame],
        dirty_swaps: List[VisionFrame],
        depth_maps: List[numpy.ndarray]
    ) -> List[VisionFrame]:
        """Process a chunk of frames with temporal smoothing."""
        alpha_maps = [self._compute_alpha(dm) for dm in depth_maps]
        smoothed_alphas = self._apply_temporal_smoothing(alpha_maps)
        
        results = []
        for i in range(len(frames)):
            result = self._composite(frames[i], dirty_swaps[i], smoothed_alphas[i])
            results.append(result)
        
        return results
    
    def _compute_alpha(self, depth_map: numpy.ndarray) -> numpy.ndarray:
        """Compute alpha mask from depth map."""
        if depth_map.max() > 1:
            depth_normalized = depth_map / 255.0
        else:
            depth_normalized = depth_map.astype(numpy.float32)
        
        _, mask = cv2.threshold(
            depth_normalized,
            self.depth_threshold,
            1.0,
            cv2.THRESH_BINARY
        )
        
        alpha = cv2.GaussianBlur(mask, self.blur_strength, 0)
        return numpy.clip(alpha, 0, 1)
    
    def _apply_temporal_smoothing(
        self,
        alpha_maps: List[numpy.ndarray]
    ) -> List[numpy.ndarray]:
        """Apply temporal smoothing to alpha maps for flicker reduction.
        
        Uses a sliding window average to smooth alpha values across frames.
        
        Args:
            alpha_maps: List of alpha maps for each frame
        
        Returns:
            Temporally smoothed alpha maps
        """
        if len(alpha_maps) <= 1:
            return alpha_maps
        
        smoothed = []
        half_window = self.temporal_window // 2
        
        for i in range(len(alpha_maps)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(alpha_maps), i + half_window + 1)
            
            window_alphas = alpha_maps[start_idx:end_idx]
            avg_alpha = numpy.mean(window_alphas, axis=0)
            smoothed.append(avg_alpha.astype(numpy.float32))
        
        return smoothed
    
    def _composite(
        self,
        original: VisionFrame,
        dirty_swap: VisionFrame,
        alpha: numpy.ndarray
    ) -> VisionFrame:
        """Apply the mayonnaise compositing formula.
        
        Formula: Final = (Dirty_Swap * (1 - Alpha)) + (Original * Alpha)
        """
        if original.shape[:2] != dirty_swap.shape[:2]:
            dirty_swap = cv2.resize(dirty_swap, (original.shape[1], original.shape[0]))
        
        if alpha.shape[:2] != original.shape[:2]:
            alpha = cv2.resize(alpha, (original.shape[1], original.shape[0]))
        
        if len(alpha.shape) == 2:
            alpha = alpha[:, :, numpy.newaxis]
        
        original_f = original.astype(numpy.float32)
        dirty_swap_f = dirty_swap.astype(numpy.float32)
        
        result = dirty_swap_f * (1.0 - alpha) + original_f * alpha
        return numpy.clip(result, 0, 255).astype(numpy.uint8)
    
    def compute_temporal_consistency(
        self,
        composited_frames: List[VisionFrame]
    ) -> float:
        """Compute temporal consistency score for quality assessment.
        
        Measures frame-to-frame similarity to detect flicker.
        Score >0.9 indicates good temporal consistency.
        
        Returns:
            Consistency score [0, 1]
        """
        if len(composited_frames) < 2:
            return 1.0
        
        similarities = []
        for i in range(1, len(composited_frames)):
            prev_gray = cv2.cvtColor(composited_frames[i - 1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(composited_frames[i], cv2.COLOR_RGB2GRAY)
            
            prev_gray = prev_gray.astype(numpy.float32) / 255.0
            curr_gray = curr_gray.astype(numpy.float32) / 255.0
            
            diff = numpy.abs(prev_gray - curr_gray)
            similarity = 1.0 - numpy.mean(diff)
            similarities.append(similarity)
        
        return float(numpy.mean(similarities))
    
    def reset(self):
        """Reset internal buffers."""
        self._alpha_buffer.clear()


def create_transparency_handler(
    depth_threshold: float = 0.74,
    blur_strength: Tuple[int, int] = (5, 5),
    temporal_window: int = 5
) -> TransparencyHandler:
    """Factory function to create TransparencyHandler."""
    return TransparencyHandler(
        depth_threshold=depth_threshold,
        blur_strength=blur_strength,
        temporal_window=temporal_window
    )


def composite_frame(
    original: VisionFrame,
    dirty_swap: VisionFrame,
    depth_map: numpy.ndarray,
    depth_threshold: float = 0.74
) -> VisionFrame:
    """Convenience function for single-frame compositing."""
    handler = TransparencyHandler(depth_threshold=depth_threshold)
    return handler.process_frame(original, dirty_swap, depth_map)


# Processor interface stubs (required for module loading)
def get_inference_pool():
    return None

def clear_inference_pool():
    pass

def register_args(program):
    pass

def apply_args(args, state_apply):
    pass

def pre_check():
    return True

def pre_process(mode):
    pass

def post_process():
    pass

def get_reference_frame(source_face, target_face, temp_frame):
    return temp_frame

def process_frame(inputs):
    return inputs.get('target_frame')

def process_frames(source_paths, queue_payloads, update_progress):
    pass

def process_image(source_paths, target_path, output_path):
    pass

def process_video(source_paths, temp_frame_paths):
    pass
