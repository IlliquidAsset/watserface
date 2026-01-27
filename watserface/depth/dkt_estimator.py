"""DKT Estimator for point tracking through occlusions.

This module provides the DKTEstimator class for tracking facial points
through semi-transparent occlusions (mayo, glasses, steam) using CoTracker3
or TAPIR point tracking models.
"""
from typing import Any, List, Optional, Tuple, Union
import numpy
import cv2

from watserface.types import VisionFrame


class DKTEstimator:
    """Point tracker for tracking facial landmarks through occlusions.
    
    Uses CoTracker3 (default) or TAPIR for robust point tracking that
    maintains spatial coherence even through semi-transparent occlusions.
    
    Attributes:
        model_type: The tracking model to use ('cotracker3' or 'tapir')
        model: The loaded tracking model
        device: The compute device (cuda/mps/cpu)
        loaded: Whether the model is loaded
    """
    
    SUPPORTED_MODELS = ['cotracker3', 'tapir']
    
    def __init__(self, model_type: str = 'cotracker3'):
        """Initialize the DKT Estimator.
        
        Args:
            model_type: The tracking model to use. Options:
                - 'cotracker3': Facebook's CoTracker3 (recommended, best accuracy)
                - 'tapir': Google's TAPIR (Apache 2.0 license, faster)
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")
        
        self.model_type = model_type
        self.model = None
        self.device = None
        self.loaded = False
    
    def load(self) -> bool:
        """Load the point tracking model.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            import torch
            
            # Auto-detect device
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            
            if self.model_type == 'cotracker3':
                self.model = self._load_cotracker3()
            elif self.model_type == 'tapir':
                self.model = self._load_tapir()
            
            if self.model is not None:
                self.loaded = True
                return True
            return False
            
        except ImportError as e:
            print(f'Required packages not installed: {e}')
            return False
        except Exception as e:
            print(f'Failed to load {self.model_type} model: {e}')
            return False
    
    def _load_cotracker3(self) -> Any:
        """Load CoTracker3 model from torch hub."""
        import torch
        try:
            model = torch.hub.load(
                "facebookresearch/co-tracker",
                "cotracker3_offline",
                trust_repo=True
            )
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f'Failed to load CoTracker3: {e}')
            return None
    
    def _load_tapir(self) -> Any:
        """Load TAPIR model from torch hub."""
        import torch
        try:
            model = torch.hub.load(
                "google-deepmind/tapnet",
                "tapir_model",
                trust_repo=True
            )
            model = model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f'Failed to load TAPIR: {e}')
            return None
    
    def track_points(
        self,
        frames: Union[List[VisionFrame], numpy.ndarray],
        query_points: Optional[numpy.ndarray] = None,
        grid_size: int = 10
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Track points across video frames.
        
        Args:
            frames: List of video frames or numpy array of shape (T, H, W, C)
            query_points: Optional (N, 2) array of (x, y) coordinates to track.
                         If None, uses a grid of grid_size x grid_size points.
            grid_size: Size of point grid if query_points not provided.
        
        Returns:
            Tuple of:
                - tracks: (T, N, 2) array of tracked point positions
                - visibility: (T, N) array of visibility scores [0, 1]
        """
        if not self.loaded:
            if not self.load():
                return self._fallback_track(frames, query_points, grid_size)
        
        try:
            import torch
            
            # Convert frames to tensor
            if isinstance(frames, list):
                frames_arr = numpy.stack(frames, axis=0)
            else:
                frames_arr = frames
            
            # Ensure RGB format
            if frames_arr.shape[-1] == 3:
                video = torch.tensor(frames_arr).permute(0, 3, 1, 2)
            else:
                video = torch.tensor(frames_arr)
            
            video = video[None].float().to(self.device)
            
            with torch.no_grad():
                if self.model_type == 'cotracker3':
                    tracks, visibility = self._track_cotracker3(
                        video, query_points, grid_size
                    )
                elif self.model_type == 'tapir':
                    tracks, visibility = self._track_tapir(
                        video, query_points, grid_size
                    )
                else:
                    return self._fallback_track(frames, query_points, grid_size)
            
            return tracks, visibility
            
        except Exception as e:
            print(f'Tracking failed: {e}')
            return self._fallback_track(frames, query_points, grid_size)
    
    def _track_cotracker3(
        self,
        video: Any,
        query_points: Optional[numpy.ndarray],
        grid_size: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Track points using CoTracker3."""
        import torch
        
        if query_points is not None:
            # Convert query points to CoTracker format
            # CoTracker expects (B, N, 3) with (t, x, y)
            n_points = query_points.shape[0]
            queries = torch.zeros((1, n_points, 3), device=self.device)
            queries[0, :, 0] = 0  # All start at frame 0
            queries[0, :, 1:] = torch.tensor(query_points, device=self.device)
            
            pred_tracks, pred_visibility = self.model(
                video,
                queries=queries
            )
        else:
            # Use automatic grid
            pred_tracks, pred_visibility = self.model(
                video,
                grid_size=grid_size
            )
        
        # Convert to numpy: (B, T, N, 2) -> (T, N, 2)
        tracks = pred_tracks[0].cpu().numpy()
        visibility = pred_visibility[0].cpu().numpy()
        
        # Squeeze visibility if needed
        if len(visibility.shape) == 3:
            visibility = visibility.squeeze(-1)
        
        return tracks, visibility
    
    def _track_tapir(
        self,
        video: Any,
        query_points: Optional[numpy.ndarray],
        grid_size: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Track points using TAPIR."""
        import torch
        
        T, _, H, W = video.shape[1:]
        
        if query_points is not None:
            # TAPIR expects (B, N, 3) with (frame_idx, x, y)
            n_points = query_points.shape[0]
            queries = torch.zeros((1, n_points, 3), device=self.device)
            queries[0, :, 0] = 0  # All start at frame 0
            queries[0, :, 1:] = torch.tensor(query_points, device=self.device)
        else:
            # Generate grid of query points
            xs = torch.linspace(0, W - 1, grid_size, device=self.device)
            ys = torch.linspace(0, H - 1, grid_size, device=self.device)
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
            
            n_points = points.shape[0]
            queries = torch.zeros((1, n_points, 3), device=self.device)
            queries[0, :, 0] = 0
            queries[0, :, 1:] = points
        
        pred_tracks, pred_visibility = self.model(
            video,
            query_points=queries,
            query_mode='first_frame'
        )
        
        # Convert to numpy
        tracks = pred_tracks[0].cpu().numpy()
        visibility = pred_visibility[0].cpu().numpy()
        
        if len(visibility.shape) == 3:
            visibility = visibility.squeeze(-1)
        
        return tracks, visibility
    
    def _fallback_track(
        self,
        frames: Union[List[VisionFrame], numpy.ndarray],
        query_points: Optional[numpy.ndarray],
        grid_size: int
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """Fallback tracking using optical flow when model unavailable."""
        if isinstance(frames, list):
            frames_arr = numpy.stack(frames, axis=0)
        else:
            frames_arr = frames
        
        T, H, W = frames_arr.shape[:3]
        
        # Generate query points if not provided
        if query_points is None:
            xs = numpy.linspace(0, W - 1, grid_size)
            ys = numpy.linspace(0, H - 1, grid_size)
            grid_x, grid_y = numpy.meshgrid(xs, ys)
            query_points = numpy.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)
        
        N = query_points.shape[0]
        tracks = numpy.zeros((T, N, 2), dtype=numpy.float32)
        visibility = numpy.ones((T, N), dtype=numpy.float32)
        
        # Initialize with query points
        tracks[0] = query_points
        
        # Simple optical flow tracking (fallback)
        for t in range(1, T):
            prev_gray = cv2.cvtColor(frames_arr[t - 1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames_arr[t], cv2.COLOR_RGB2GRAY)
            
            # Track points using Lucas-Kanade
            prev_pts = tracks[t - 1].reshape(-1, 1, 2).astype(numpy.float32)
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts, None
            )
            
            if curr_pts is not None:
                tracks[t] = curr_pts.reshape(-1, 2)
                visibility[t] = status.flatten().astype(numpy.float32)
            else:
                tracks[t] = tracks[t - 1]
                visibility[t] = 0.0
        
        return tracks, visibility
    
    def estimate_alpha(
        self,
        frame: VisionFrame,
        depth_map: numpy.ndarray,
        threshold: float = 0.74,
        blur_strength: Tuple[int, int] = (5, 5)
    ) -> numpy.ndarray:
        """Generate alpha mask from depth map using mayonnaise strategy threshold.
        
        Args:
            frame: Input frame (used for shape reference)
            depth_map: Normalized depth map [0, 1]
            threshold: Depth threshold for occlusion detection (default 0.74 = 188/255)
            blur_strength: Gaussian blur kernel size for mask softening
        
        Returns:
            Alpha mask [0, 1] where 1 = occlusion (foreground)
        """
        # Normalize depth map
        if depth_map.max() > 1:
            depth_normalized = depth_map / 255.0
        else:
            depth_normalized = depth_map
        
        # Threshold to create occlusion mask
        # Higher depth values = closer to camera = occlusion
        _, occlusion_mask = cv2.threshold(
            depth_normalized.astype(numpy.float32),
            threshold,
            1.0,
            cv2.THRESH_BINARY
        )
        
        # Soften mask edges
        alpha = cv2.GaussianBlur(occlusion_mask, blur_strength, 0)
        
        # Ensure proper range
        alpha = numpy.clip(alpha, 0, 1)
        
        return alpha
    
    def estimate_alpha_soft(
        self,
        depth_map: numpy.ndarray,
        threshold: float = 0.74,
        smoothness: float = 0.05
    ) -> numpy.ndarray:
        """Generate soft alpha mask with gradient transition.
        
        Args:
            depth_map: Normalized depth map [0, 1]
            threshold: Center of transition
            smoothness: Width of transition region
        
        Returns:
            Soft alpha mask [0, 1]
        """
        if depth_map.max() > 1:
            depth_normalized = depth_map / 255.0
        else:
            depth_normalized = depth_map
        
        # Create soft transition around threshold
        alpha = numpy.clip(
            (depth_normalized - threshold + smoothness / 2) / smoothness,
            0, 1
        )
        
        return alpha


# Module-level singleton for convenience
_estimator: Optional[DKTEstimator] = None


def get_dkt_estimator(model_type: str = 'cotracker3') -> DKTEstimator:
    """Get or create a DKT estimator singleton.
    
    Args:
        model_type: The tracking model to use
    
    Returns:
        DKTEstimator instance
    """
    global _estimator
    
    if _estimator is None or _estimator.model_type != model_type:
        _estimator = DKTEstimator(model_type)
    
    return _estimator


def track_points(
    frames: Union[List[VisionFrame], numpy.ndarray],
    query_points: Optional[numpy.ndarray] = None,
    model_type: str = 'cotracker3',
    grid_size: int = 10
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Convenience function to track points across frames.
    
    Args:
        frames: Video frames
        query_points: Points to track (optional)
        model_type: Tracking model to use
        grid_size: Grid size if query_points not provided
    
    Returns:
        Tuple of (tracks, visibility)
    """
    estimator = get_dkt_estimator(model_type)
    return estimator.track_points(frames, query_points, grid_size)


def estimate_alpha(
    frame: VisionFrame,
    depth_map: numpy.ndarray,
    threshold: float = 0.74
) -> numpy.ndarray:
    """Convenience function to estimate alpha from depth.
    
    Args:
        frame: Input frame
        depth_map: Depth map
        threshold: Depth threshold
    
    Returns:
        Alpha mask
    """
    estimator = get_dkt_estimator()
    return estimator.estimate_alpha(frame, depth_map, threshold)
