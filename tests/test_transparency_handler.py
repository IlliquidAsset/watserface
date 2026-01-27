"""Tests for TransparencyHandler - video transparency compositing with temporal coherence."""
import numpy
import pytest


class TestTransparencyHandlerInitialization:
    """Test TransparencyHandler initialization."""
    
    def test_class_exists(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        assert TransparencyHandler is not None
    
    def test_initializes_with_defaults(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        assert handler is not None
        assert handler.depth_threshold == 0.74
        assert handler.blur_strength == (5, 5)
        assert handler.temporal_window == 5
    
    def test_accepts_custom_parameters(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler(
            depth_threshold=0.8,
            blur_strength=(7, 7),
            temporal_window=10
        )
        assert handler.depth_threshold == 0.8
        assert handler.blur_strength == (7, 7)
        assert handler.temporal_window == 10


class TestTransparencyHandlerProcessFrame:
    """Test single-frame processing."""
    
    def test_has_process_frame_method(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        assert hasattr(handler, 'process_frame')
        assert callable(handler.process_frame)
    
    def test_process_frame_returns_correct_shape(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        
        h, w = 100, 100
        original = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        dirty_swap = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
        depth_map = numpy.zeros((h, w), dtype=numpy.float32)
        
        result = handler.process_frame(original, dirty_swap, depth_map)
        
        assert result.shape == (h, w, 3)
        assert result.dtype == numpy.uint8
    
    def test_process_frame_compositing_formula(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler(depth_threshold=0.5, blur_strength=(1, 1))
        
        h, w = 10, 10
        original = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        dirty_swap = numpy.ones((h, w, 3), dtype=numpy.uint8) * 200
        
        depth_map = numpy.zeros((h, w), dtype=numpy.float32)
        depth_map[:5, :] = 0.3
        depth_map[5:, :] = 0.7
        
        result = handler.process_frame(original, dirty_swap, depth_map)
        
        assert result[:3, :].mean() > 100
        assert result[7:, :].mean() < 100


class TestTransparencyHandlerProcessVideo:
    """Test video processing with temporal coherence."""
    
    def test_has_process_video_method(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        assert hasattr(handler, 'process_video')
        assert callable(handler.process_video)
    
    def test_process_video_returns_correct_length(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        
        num_frames = 10
        h, w = 50, 50
        
        frames = [numpy.zeros((h, w, 3), dtype=numpy.uint8) for _ in range(num_frames)]
        dirty_swaps = [numpy.ones((h, w, 3), dtype=numpy.uint8) * 255 for _ in range(num_frames)]
        depth_maps = [numpy.random.rand(h, w).astype(numpy.float32) for _ in range(num_frames)]
        
        results = handler.process_video(frames, dirty_swaps, depth_maps)
        
        assert len(results) == num_frames
    
    def test_process_video_validates_lengths(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        
        frames = [numpy.zeros((50, 50, 3), dtype=numpy.uint8) for _ in range(5)]
        dirty_swaps = [numpy.zeros((50, 50, 3), dtype=numpy.uint8) for _ in range(3)]
        depth_maps = [numpy.zeros((50, 50), dtype=numpy.float32) for _ in range(5)]
        
        with pytest.raises(ValueError):
            handler.process_video(frames, dirty_swaps, depth_maps)
    
    def test_process_video_handles_empty_input(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        
        results = handler.process_video([], [], [])
        assert results == []


class TestTransparencyHandlerTemporalSmoothing:
    """Test temporal smoothing functionality."""
    
    def test_has_temporal_smoothing_method(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        assert hasattr(handler, '_apply_temporal_smoothing')
    
    def test_temporal_smoothing_reduces_variance(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler(temporal_window=5)
        
        alpha_maps = []
        for i in range(10):
            alpha = numpy.ones((20, 20), dtype=numpy.float32) * (0.3 + 0.4 * (i % 2))
            alpha_maps.append(alpha)
        
        original_variance = numpy.var([a.mean() for a in alpha_maps])
        
        smoothed = handler._apply_temporal_smoothing(alpha_maps)
        smoothed_variance = numpy.var([a.mean() for a in smoothed])
        
        assert smoothed_variance < original_variance
    
    def test_temporal_smoothing_preserves_length(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler(temporal_window=3)
        
        alpha_maps = [numpy.random.rand(20, 20).astype(numpy.float32) for _ in range(10)]
        smoothed = handler._apply_temporal_smoothing(alpha_maps)
        
        assert len(smoothed) == len(alpha_maps)


class TestTransparencyHandlerTemporalConsistency:
    """Test temporal consistency measurement."""
    
    def test_has_compute_temporal_consistency(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        assert hasattr(handler, 'compute_temporal_consistency')
    
    def test_consistent_frames_high_score(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        
        frame = numpy.ones((50, 50, 3), dtype=numpy.uint8) * 128
        frames = [frame.copy() for _ in range(10)]
        
        score = handler.compute_temporal_consistency(frames)
        assert score > 0.99
    
    def test_varying_frames_lower_score(self):
        from watserface.processors.modules.transparency_handler import TransparencyHandler
        handler = TransparencyHandler()
        
        frames = []
        for i in range(10):
            frame = numpy.ones((50, 50, 3), dtype=numpy.uint8) * (50 + i * 20)
            frames.append(frame)
        
        score = handler.compute_temporal_consistency(frames)
        assert 0 < score < 0.99


class TestTransparencyHandlerFactoryFunctions:
    """Test factory and convenience functions."""
    
    def test_create_transparency_handler_exists(self):
        from watserface.processors.modules.transparency_handler import create_transparency_handler
        assert create_transparency_handler is not None
    
    def test_create_transparency_handler_returns_instance(self):
        from watserface.processors.modules.transparency_handler import (
            create_transparency_handler,
            TransparencyHandler
        )
        handler = create_transparency_handler()
        assert isinstance(handler, TransparencyHandler)
    
    def test_composite_frame_function_exists(self):
        from watserface.processors.modules.transparency_handler import composite_frame
        assert composite_frame is not None
    
    def test_composite_frame_works(self):
        from watserface.processors.modules.transparency_handler import composite_frame
        
        h, w = 50, 50
        original = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        dirty_swap = numpy.ones((h, w, 3), dtype=numpy.uint8) * 255
        depth_map = numpy.zeros((h, w), dtype=numpy.float32)
        
        result = composite_frame(original, dirty_swap, depth_map)
        
        assert result is not None
        assert result.shape == (h, w, 3)
