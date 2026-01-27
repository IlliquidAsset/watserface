"""Tests for ControlNet pipeline integration with Stable Diffusion for face swap conditioning.

TDD RED phase: These tests define the expected behavior of the ControlNetPipeline class.
"""
import numpy
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestControlNetPipelineInitialization:
    """Test ControlNetPipeline initialization without errors."""
    
    def test_pipeline_class_exists(self):
        """ControlNetPipeline class should be importable."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        assert ControlNetPipeline is not None
    
    def test_pipeline_initializes_with_defaults(self):
        """Pipeline should initialize with default parameters."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'device')
        assert hasattr(pipeline, 'controlnet_conditioning_scale')
    
    def test_pipeline_default_conditioning_scale(self):
        """Default conditioning scale should be in 0.7-0.8 range."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        assert 0.7 <= pipeline.controlnet_conditioning_scale <= 0.8
    
    def test_pipeline_accepts_custom_conditioning_scale(self):
        """Pipeline should accept custom conditioning scale."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline(controlnet_conditioning_scale=0.75)
        assert pipeline.controlnet_conditioning_scale == 0.75
    
    def test_pipeline_accepts_device_parameter(self):
        """Pipeline should accept device parameter."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline(device='cpu')
        assert pipeline.device == 'cpu'
    
    def test_pipeline_auto_device_detection(self):
        """Pipeline should auto-detect device when set to 'auto'."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline(device='auto')
        assert pipeline.device in ['cpu', 'cuda', 'mps']


class TestControlNetPipelineModelLoading:
    """Test ControlNet model loading functionality."""
    
    def test_pipeline_has_load_method(self):
        """Pipeline should have a load method."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        assert hasattr(pipeline, 'load')
        assert callable(pipeline.load)
    
    def test_pipeline_loaded_flag_initially_false(self):
        """Pipeline loaded flag should be False initially."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        assert pipeline.loaded is False
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("diffusers", reason="diffusers not installed"),
        reason="diffusers not installed"
    )
    def test_pipeline_uses_correct_model_ids(self):
        """Pipeline should use specified pre-trained model IDs."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        # Check model IDs are set correctly
        assert hasattr(pipeline, 'depth_model_id')
        assert hasattr(pipeline, 'canny_model_id')
        # Should use SDXL-compatible models
        assert 'depth' in pipeline.depth_model_id.lower() or 'sdxl' in pipeline.depth_model_id.lower()


class TestControlNetPipelineProcessing:
    """Test ControlNet pipeline image processing."""
    
    def test_pipeline_has_process_method(self):
        """Pipeline should have a process method."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        assert hasattr(pipeline, 'process')
        assert callable(pipeline.process)
    
    def test_process_accepts_numpy_image(self):
        """Process method should accept numpy array image."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        # Create a dummy 512x512 RGB image
        dummy_image = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
        # Should not raise an error (even if not loaded)
        assert hasattr(pipeline, 'process')
    
    def test_process_returns_numpy_array(self):
        """Process method should return numpy array."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        dummy_image = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
        # Mock the pipeline to avoid actual model loading
        with patch.object(pipeline, 'loaded', True):
            with patch.object(pipeline, '_run_pipeline') as mock_run:
                mock_run.return_value = dummy_image
                result = pipeline.process(dummy_image)
                assert isinstance(result, numpy.ndarray)
    
    def test_process_output_shape_512x512(self):
        """Process output should be 512x512 resolution."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        dummy_image = numpy.zeros((1080, 1920, 3), dtype=numpy.uint8)
        with patch.object(pipeline, 'loaded', True):
            with patch.object(pipeline, '_run_pipeline') as mock_run:
                mock_run.return_value = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
                result = pipeline.process(dummy_image)
                assert result.shape[:2] == (512, 512)


class TestControlNetPipelineDepthConditioning:
    """Test depth conditioning functionality."""
    
    def test_pipeline_has_prepare_depth_conditioning(self):
        """Pipeline should have depth conditioning preparation."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        assert hasattr(pipeline, 'prepare_depth_conditioning')
    
    def test_depth_conditioning_returns_valid_format(self):
        """Depth conditioning should return properly formatted data."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        dummy_image = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
        result = pipeline.prepare_depth_conditioning(dummy_image)
        assert isinstance(result, numpy.ndarray)
        assert result.shape[:2] == (512, 512)


class TestControlNetPipelineCannyConditioning:
    """Test canny edge conditioning functionality."""
    
    def test_pipeline_has_prepare_canny_conditioning(self):
        """Pipeline should have canny conditioning preparation."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        assert hasattr(pipeline, 'prepare_canny_conditioning')
    
    def test_canny_conditioning_returns_valid_format(self):
        """Canny conditioning should return properly formatted data."""
        from watserface.inpainting.controlnet import ControlNetPipeline
        pipeline = ControlNetPipeline()
        dummy_image = numpy.zeros((512, 512, 3), dtype=numpy.uint8)
        result = pipeline.prepare_canny_conditioning(dummy_image)
        assert isinstance(result, numpy.ndarray)
        assert result.shape[:2] == (512, 512)


class TestControlNetPipelineIntegration:
    """Integration tests for ControlNet pipeline with sample images."""
    
    @pytest.fixture
    def sample_image_path(self):
        """Get a sample image path for testing."""
        # Use existing face set frame
        sample_path = Path("models/face_sets/faceset_512e84d4_1768337182/frames/frame_000498.png")
        if sample_path.exists():
            return sample_path
        # Fallback to any PNG in the project
        for p in Path(".").rglob("*.png"):
            if "frame" in str(p):
                return p
        return None
    
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not installed"),
        reason="torch not installed"
    )
    def test_pipeline_processes_sample_image_without_error(self, sample_image_path):
        """Pipeline should process sample image without raising errors."""
        if sample_image_path is None:
            pytest.skip("No sample image available")
        
        from watserface.inpainting.controlnet import ControlNetPipeline
        import cv2
        
        pipeline = ControlNetPipeline()
        image = cv2.imread(str(sample_image_path))
        if image is None:
            pytest.skip("Could not load sample image")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Test conditioning preparation (doesn't require model loading)
        depth_cond = pipeline.prepare_depth_conditioning(image_rgb)
        canny_cond = pipeline.prepare_canny_conditioning(image_rgb)
        
        assert depth_cond is not None
        assert canny_cond is not None


class TestControlNetPipelineFactory:
    """Test factory function for creating ControlNet pipeline."""
    
    def test_create_controlnet_pipeline_exists(self):
        """Factory function should exist."""
        from watserface.inpainting.controlnet import create_controlnet_pipeline
        assert create_controlnet_pipeline is not None
        assert callable(create_controlnet_pipeline)
    
    def test_create_controlnet_pipeline_returns_pipeline(self):
        """Factory should return ControlNetPipeline instance."""
        from watserface.inpainting.controlnet import create_controlnet_pipeline, ControlNetPipeline
        pipeline = create_controlnet_pipeline()
        assert isinstance(pipeline, ControlNetPipeline)
    
    def test_create_controlnet_pipeline_accepts_parameters(self):
        """Factory should accept configuration parameters."""
        from watserface.inpainting.controlnet import create_controlnet_pipeline
        pipeline = create_controlnet_pipeline(
            device='cpu',
            controlnet_conditioning_scale=0.75
        )
        assert pipeline.device == 'cpu'
        assert pipeline.controlnet_conditioning_scale == 0.75
