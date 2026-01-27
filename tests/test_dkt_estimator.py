"""Tests for DKTEstimator class - point tracking through occlusions.

TDD tests for the DKTEstimator class that tracks facial points
through semi-transparent occlusions using CoTracker3 or TAPIR.
"""
import numpy
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestDKTEstimatorInitialization:
    """Test DKTEstimator class initialization."""
    
    def test_class_exists(self):
        """DKTEstimator class should be importable."""
        from watserface.depth.dkt_estimator import DKTEstimator
        assert DKTEstimator is not None
    
    def test_initializes_with_defaults(self):
        """DKTEstimator should initialize with default parameters."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        assert estimator is not None
        assert estimator.model_type == 'cotracker3'
        assert estimator.loaded is False
    
    def test_accepts_cotracker3_model_type(self):
        """DKTEstimator should accept cotracker3 model type."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator(model_type='cotracker3')
        assert estimator.model_type == 'cotracker3'
    
    def test_accepts_tapir_model_type(self):
        """DKTEstimator should accept tapir model type."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator(model_type='tapir')
        assert estimator.model_type == 'tapir'
    
    def test_rejects_invalid_model_type(self):
        """DKTEstimator should reject invalid model types."""
        from watserface.depth.dkt_estimator import DKTEstimator
        with pytest.raises(ValueError):
            DKTEstimator(model_type='invalid_model')


class TestDKTEstimatorLoad:
    """Test DKTEstimator model loading."""
    
    def test_has_load_method(self):
        """DKTEstimator should have a load method."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        assert hasattr(estimator, 'load')
        assert callable(estimator.load)
    
    def test_loaded_flag_false_initially(self):
        """Loaded flag should be False before load() is called."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        assert estimator.loaded is False
    
    @patch('torch.hub.load')
    @patch('torch.backends.mps.is_available', return_value=False)
    @patch('torch.cuda.is_available', return_value=False)
    def test_load_sets_device(self, mock_cuda, mock_mps, mock_hub):
        """Load should auto-detect device."""
        from watserface.depth.dkt_estimator import DKTEstimator
        
        mock_model = MagicMock()
        mock_hub.return_value = mock_model
        
        estimator = DKTEstimator()
        estimator.load()
        
        assert estimator.device == 'cpu'


class TestDKTEstimatorTrackPoints:
    """Test point tracking functionality."""
    
    def test_has_track_points_method(self):
        """DKTEstimator should have a track_points method."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        assert hasattr(estimator, 'track_points')
        assert callable(estimator.track_points)
    
    def test_track_points_accepts_frames_list(self):
        """track_points should accept a list of frames."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        
        # Create dummy frames
        frames = [numpy.zeros((100, 100, 3), dtype=numpy.uint8) for _ in range(5)]
        
        # Should not raise
        tracks, visibility = estimator.track_points(frames)
        assert tracks is not None
        assert visibility is not None
    
    def test_track_points_accepts_numpy_array(self):
        """track_points should accept a numpy array of frames."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        
        # Create dummy frames as numpy array (T, H, W, C)
        frames = numpy.zeros((5, 100, 100, 3), dtype=numpy.uint8)
        
        tracks, visibility = estimator.track_points(frames)
        assert tracks is not None
        assert visibility is not None
    
    def test_track_points_returns_correct_shapes(self):
        """track_points should return correctly shaped arrays."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        
        T, H, W = 5, 100, 100
        grid_size = 4
        
        frames = numpy.zeros((T, H, W, 3), dtype=numpy.uint8)
        tracks, visibility = estimator.track_points(frames, grid_size=grid_size)
        
        # tracks should be (T, N, 2) where N = grid_size^2
        expected_n = grid_size * grid_size
        assert tracks.shape == (T, expected_n, 2)
        
        # visibility should be (T, N)
        assert visibility.shape == (T, expected_n)
    
    def test_track_points_with_query_points(self):
        """track_points should accept custom query points."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        
        T, H, W = 5, 100, 100
        frames = numpy.zeros((T, H, W, 3), dtype=numpy.uint8)
        
        # Custom query points
        query_points = numpy.array([
            [25, 25],
            [50, 50],
            [75, 75]
        ], dtype=numpy.float32)
        
        tracks, visibility = estimator.track_points(frames, query_points=query_points)
        
        # Should track the 3 specified points
        assert tracks.shape == (T, 3, 2)
        assert visibility.shape == (T, 3)


class TestDKTEstimatorEstimateAlpha:
    """Test alpha mask estimation from depth."""
    
    def test_has_estimate_alpha_method(self):
        """DKTEstimator should have estimate_alpha method."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        assert hasattr(estimator, 'estimate_alpha')
        assert callable(estimator.estimate_alpha)
    
    def test_estimate_alpha_returns_valid_mask(self):
        """estimate_alpha should return a valid alpha mask."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        
        # Create dummy frame and depth map
        frame = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        depth_map = numpy.random.rand(100, 100).astype(numpy.float32)
        
        alpha = estimator.estimate_alpha(frame, depth_map)
        
        assert alpha is not None
        assert alpha.shape == (100, 100)
        assert alpha.min() >= 0
        assert alpha.max() <= 1
    
    def test_estimate_alpha_respects_threshold(self):
        """estimate_alpha should respect the threshold parameter."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        
        frame = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        
        # Create depth map with clear regions
        depth_map = numpy.zeros((100, 100), dtype=numpy.float32)
        depth_map[:50, :] = 0.5  # Below threshold
        depth_map[50:, :] = 0.9  # Above threshold (0.74)
        
        alpha = estimator.estimate_alpha(frame, depth_map, threshold=0.74)
        
        # Lower region should have low alpha (below threshold)
        assert alpha[:40, :].mean() < 0.5
        
        # Upper region should have high alpha (above threshold)
        assert alpha[60:, :].mean() > 0.5
    
    def test_estimate_alpha_handles_uint8_depth(self):
        """estimate_alpha should handle uint8 depth maps."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        
        frame = numpy.zeros((100, 100, 3), dtype=numpy.uint8)
        depth_map = numpy.random.randint(0, 256, (100, 100), dtype=numpy.uint8)
        
        alpha = estimator.estimate_alpha(frame, depth_map)
        
        assert alpha is not None
        assert alpha.min() >= 0
        assert alpha.max() <= 1


class TestDKTEstimatorConvenienceFunctions:
    """Test module-level convenience functions."""
    
    def test_get_dkt_estimator_exists(self):
        """get_dkt_estimator function should exist."""
        from watserface.depth.dkt_estimator import get_dkt_estimator
        assert get_dkt_estimator is not None
        assert callable(get_dkt_estimator)
    
    def test_get_dkt_estimator_returns_estimator(self):
        """get_dkt_estimator should return DKTEstimator instance."""
        from watserface.depth.dkt_estimator import get_dkt_estimator, DKTEstimator
        estimator = get_dkt_estimator()
        assert isinstance(estimator, DKTEstimator)
    
    def test_track_points_function_exists(self):
        """Module-level track_points function should exist."""
        from watserface.depth.dkt_estimator import track_points
        assert track_points is not None
        assert callable(track_points)
    
    def test_estimate_alpha_function_exists(self):
        """Module-level estimate_alpha function should exist."""
        from watserface.depth.dkt_estimator import estimate_alpha
        assert estimate_alpha is not None
        assert callable(estimate_alpha)


class TestDKTEstimatorFallback:
    """Test fallback behavior when models unavailable."""
    
    def test_fallback_track_works_without_model(self):
        """Fallback tracking should work when model unavailable."""
        from watserface.depth.dkt_estimator import DKTEstimator
        estimator = DKTEstimator()
        
        # Don't load model - should use fallback
        frames = numpy.zeros((5, 100, 100, 3), dtype=numpy.uint8)
        
        # Add some visual features for optical flow
        for i, frame in enumerate(frames):
            cv_i = i * 10
            frame[40:60, 40 + cv_i:60 + cv_i] = 255
        
        tracks, visibility = estimator.track_points(frames, grid_size=3)
        
        assert tracks is not None
        assert visibility is not None
        assert tracks.shape[0] == 5  # T frames
        assert tracks.shape[2] == 2  # (x, y) coordinates


# Import cv2 for fallback test
import cv2
