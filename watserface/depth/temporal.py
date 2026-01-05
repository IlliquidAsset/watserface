"""Temporal consistency for video depth estimation and alpha blending."""
from typing import List, Optional, Tuple
import cv2
import numpy

from watserface.types import VisionFrame
from watserface.depth.estimator import DepthEstimator


class TemporalDepthEstimator:
    
    def __init__(
        self,
        model_type: str = 'midas_small',
        history_size: int = 5,
        smoothing_alpha: float = 0.7
    ):
        self.estimator = DepthEstimator(model_type)
        self.history_size = history_size
        self.smoothing_alpha = smoothing_alpha
        
        self.depth_history: List[numpy.ndarray] = []
        self.frame_history: List[numpy.ndarray] = []
        self.flow_history: List[numpy.ndarray] = []
    
    def estimate_with_temporal(self, frame: VisionFrame) -> Tuple[numpy.ndarray, numpy.ndarray]:
        current_depth = self.estimator.estimate(frame)
        
        if not self.depth_history:
            self.depth_history.append(current_depth)
            self.frame_history.append(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY))
            return current_depth, numpy.ones_like(current_depth)
        
        gray_current = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray_prev = self.frame_history[-1]
        
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_current, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        warped_prev_depth = self._warp_with_flow(self.depth_history[-1], flow)
        
        occlusion_mask = self._detect_occlusions(flow, current_depth, warped_prev_depth)
        
        blended_depth = current_depth * (1 - occlusion_mask) + \
                       warped_prev_depth * occlusion_mask * self.smoothing_alpha + \
                       current_depth * occlusion_mask * (1 - self.smoothing_alpha)
        
        if len(self.depth_history) >= 3:
            weights = numpy.array([0.1, 0.2, 0.3, 0.4])[-len(self.depth_history):]
            weights = weights / weights.sum()
            
            warped_depths = [blended_depth]
            for i, hist_depth in enumerate(reversed(self.depth_history[:-1])):
                if len(warped_depths) >= len(weights):
                    break
                warped_depths.insert(0, hist_depth)
            
            if len(warped_depths) == len(weights):
                blended_depth = sum(w * d for w, d in zip(weights, warped_depths))
        
        self.depth_history.append(blended_depth)
        self.frame_history.append(gray_current)
        self.flow_history.append(flow)
        
        if len(self.depth_history) > self.history_size:
            self.depth_history.pop(0)
            self.frame_history.pop(0)
            if self.flow_history:
                self.flow_history.pop(0)
        
        confidence = 1.0 - occlusion_mask
        
        return blended_depth.astype(numpy.float32), confidence.astype(numpy.float32)
    
    def _warp_with_flow(self, depth: numpy.ndarray, flow: numpy.ndarray) -> numpy.ndarray:
        h, w = depth.shape[:2]
        
        y_coords, x_coords = numpy.mgrid[0:h, 0:w].astype(numpy.float32)
        
        new_x = x_coords + flow[:, :, 0]
        new_y = y_coords + flow[:, :, 1]
        
        warped = cv2.remap(
            depth, new_x, new_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return warped
    
    def _detect_occlusions(
        self,
        flow: numpy.ndarray,
        current_depth: numpy.ndarray,
        warped_prev_depth: numpy.ndarray
    ) -> numpy.ndarray:
        flow_magnitude = numpy.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        large_motion = flow_magnitude > numpy.percentile(flow_magnitude, 90)
        
        depth_diff = numpy.abs(current_depth - warped_prev_depth)
        depth_diff_normalized = depth_diff / (current_depth.max() - current_depth.min() + 1e-8)
        large_depth_change = depth_diff_normalized > 0.2
        
        occlusion_mask = (large_motion | large_depth_change).astype(numpy.float32)
        
        kernel = numpy.ones((5, 5), numpy.uint8)
        occlusion_mask = cv2.dilate(occlusion_mask, kernel, iterations=1)
        occlusion_mask = cv2.GaussianBlur(occlusion_mask, (11, 11), 0)
        
        return occlusion_mask
    
    def reset(self) -> None:
        self.depth_history.clear()
        self.frame_history.clear()
        self.flow_history.clear()


class AlphaEstimator:
    
    def __init__(self, edge_sensitivity: float = 0.5):
        self.edge_sensitivity = edge_sensitivity
    
    def estimate_alpha_from_depth(
        self,
        depth_map: numpy.ndarray,
        frame: VisionFrame,
        face_mask: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        depth_gradient_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        depth_gradient_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        depth_edges = numpy.sqrt(depth_gradient_x ** 2 + depth_gradient_y ** 2)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(numpy.float64)
        color_gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        color_gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        color_edges = numpy.sqrt(color_gradient_x ** 2 + color_gradient_y ** 2)
        
        depth_edges_norm = depth_edges / (depth_edges.max() + 1e-8)
        color_edges_norm = color_edges / (color_edges.max() + 1e-8)
        
        depth_only_edges = depth_edges_norm * (1 - color_edges_norm)
        depth_only_edges = numpy.clip(depth_only_edges * (1 / self.edge_sensitivity), 0, 1)
        
        alpha = numpy.ones_like(depth_map)
        
        boundary_zone = depth_only_edges > 0.1
        
        boundary_dilated = cv2.dilate(
            boundary_zone.astype(numpy.uint8),
            numpy.ones((7, 7), numpy.uint8),
            iterations=2
        )
        
        distance_to_edge = cv2.distanceTransform(
            1 - boundary_dilated,
            cv2.DIST_L2,
            5
        )
        max_dist = distance_to_edge.max() + 1e-8
        alpha = numpy.clip(distance_to_edge / max_dist, 0, 1)
        
        alpha = 1 - alpha
        alpha = cv2.GaussianBlur(alpha.astype(numpy.float32), (11, 11), 0)
        
        if face_mask is not None:
            face_mask_norm = face_mask.astype(numpy.float32)
            if face_mask_norm.max() > 1:
                face_mask_norm = face_mask_norm / 255.0
            alpha = alpha * face_mask_norm
        
        return alpha.astype(numpy.float32)
    
    def detect_translucent_objects(
        self,
        depth_map: numpy.ndarray,
        frame: VisionFrame
    ) -> Tuple[numpy.ndarray, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(numpy.float64) / 255.0
        
        depth_var = cv2.blur((depth_map ** 2), (7, 7)) - cv2.blur(depth_map, (7, 7)) ** 2
        color_var = cv2.blur((gray ** 2), (7, 7)) - cv2.blur(gray, (7, 7)) ** 2
        
        translucent_indicator = depth_var * (1 - color_var)
        
        threshold = numpy.percentile(translucent_indicator, 95)
        translucent_mask = (translucent_indicator > threshold * 0.5).astype(numpy.float32)
        
        kernel = numpy.ones((5, 5), numpy.uint8)
        translucent_mask = cv2.morphologyEx(translucent_mask, cv2.MORPH_CLOSE, kernel)
        translucent_mask = cv2.GaussianBlur(translucent_mask, (7, 7), 0)
        
        confidence = min(1.0, translucent_mask.sum() / (translucent_mask.size * 0.1))
        
        return translucent_mask, confidence


def create_temporal_estimator(
    model_type: str = 'midas_small',
    history_size: int = 5
) -> TemporalDepthEstimator:
    return TemporalDepthEstimator(model_type=model_type, history_size=history_size)


def create_alpha_estimator(edge_sensitivity: float = 0.5) -> AlphaEstimator:
    return AlphaEstimator(edge_sensitivity=edge_sensitivity)
