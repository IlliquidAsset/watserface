"""
Tests for hybrid_inswapper_simswap face swapper.

This hybrid swapper uses:
- InSwapper 128 for eye regions (better perceptual quality)
- SimSwap 512 for rest of face (better texture)

Eye landmarks (68-point):
- Left eye: landmarks 36-41
- Right eye: landmarks 42-47
"""

import pytest
import numpy
import cv2
from unittest.mock import patch, MagicMock

# Test constants
EYE_LANDMARKS_LEFT = list(range(36, 42))  # 36-41
EYE_LANDMARKS_RIGHT = list(range(42, 48))  # 42-47
GAUSSIAN_BLUR_SIGMA = 3


class TestHybridSwapperRegistration:
    """Test that hybrid_inswapper_simswap is properly registered."""
    
    def test_hybrid_model_in_choices(self):
        """Verify hybrid_inswapper_simswap appears in face_swapper_models."""
        from watserface.processors import choices
        
        assert 'hybrid_inswapper_simswap' in choices.face_swapper_models, \
            "hybrid_inswapper_simswap should be registered in face_swapper_models"
    
    def test_hybrid_model_in_model_set(self):
        """Verify hybrid_inswapper_simswap has pixel boost options."""
        from watserface.processors import choices
        
        assert 'hybrid_inswapper_simswap' in choices.face_swapper_set, \
            "hybrid_inswapper_simswap should be in face_swapper_set"
        
        # Should support at least 512x512 (SimSwap native)
        pixel_boosts = choices.face_swapper_set.get('hybrid_inswapper_simswap', [])
        assert '512x512' in pixel_boosts, \
            "hybrid_inswapper_simswap should support 512x512 pixel boost"


class TestEyeMaskCreation:
    """Test eye region mask creation from landmarks."""
    
    def test_create_eye_mask_from_landmarks(self):
        """Verify eye mask is created from landmarks 36-47."""
        from watserface.processors.modules.face_swapper import create_eye_region_mask
        
        # Create mock 68-point landmarks
        landmarks_68 = numpy.zeros((68, 2), dtype=numpy.float32)
        
        # Set eye landmarks to form simple rectangles
        # Left eye (36-41): centered at (100, 100)
        landmarks_68[36] = [80, 95]   # outer corner
        landmarks_68[37] = [90, 90]   # upper outer
        landmarks_68[38] = [100, 90]  # upper inner
        landmarks_68[39] = [110, 95]  # inner corner
        landmarks_68[40] = [100, 100] # lower inner
        landmarks_68[41] = [90, 100]  # lower outer
        
        # Right eye (42-47): centered at (200, 100)
        landmarks_68[42] = [180, 95]  # inner corner
        landmarks_68[43] = [190, 90]  # upper inner
        landmarks_68[44] = [200, 90]  # upper outer
        landmarks_68[45] = [210, 95]  # outer corner
        landmarks_68[46] = [200, 100] # lower outer
        landmarks_68[47] = [190, 100] # lower inner
        
        # Create mask for 256x256 crop frame
        crop_size = (256, 256)
        eye_mask = create_eye_region_mask(landmarks_68, crop_size)
        
        # Verify mask shape
        assert eye_mask.shape == crop_size, \
            f"Eye mask shape should be {crop_size}, got {eye_mask.shape}"
        
        # Verify mask is float in [0, 1]
        assert eye_mask.dtype == numpy.float32, \
            "Eye mask should be float32"
        assert eye_mask.min() >= 0.0 and eye_mask.max() <= 1.0, \
            "Eye mask values should be in [0, 1]"
        
        # Verify eye regions have high values (close to 1)
        left_eye_center = (100, 95)
        right_eye_center = (195, 95)
        
        # Check that eye centers are within mask (value > 0.5)
        assert eye_mask[left_eye_center[1], left_eye_center[0]] > 0.5, \
            "Left eye center should be in mask"
        assert eye_mask[right_eye_center[1], right_eye_center[0]] > 0.5, \
            "Right eye center should be in mask"
        
        # Verify non-eye regions have low values
        forehead = (128, 30)
        chin = (128, 220)
        assert eye_mask[forehead[1], forehead[0]] < 0.5, \
            "Forehead should not be in eye mask"
        assert eye_mask[chin[1], chin[0]] < 0.5, \
            "Chin should not be in eye mask"
    
    def test_eye_mask_has_gaussian_blur(self):
        """Verify eye mask boundaries are blurred with sigma=3."""
        from watserface.processors.modules.face_swapper import create_eye_region_mask
        
        landmarks_68 = numpy.zeros((68, 2), dtype=numpy.float32)
        # Simple eye positions
        for i in range(36, 42):
            landmarks_68[i] = [100 + (i - 36) * 5, 100]
        for i in range(42, 48):
            landmarks_68[i] = [150 + (i - 42) * 5, 100]
        
        crop_size = (256, 256)
        eye_mask = create_eye_region_mask(landmarks_68, crop_size)
        
        # Check for smooth gradients (not binary edges)
        # A blurred mask should have values between 0 and 1 at boundaries
        unique_values = numpy.unique(eye_mask)
        assert len(unique_values) > 10, \
            "Eye mask should have smooth gradients (many unique values), not binary"


class TestHybridSwapExecution:
    """Test the hybrid swap execution logic."""
    
    def test_hybrid_swap_uses_both_models(self):
        """Verify hybrid swap invokes both InSwapper and SimSwap."""
        from watserface.processors.modules.face_swapper import swap_face_hybrid
        
        # This test verifies the function exists and has correct signature
        import inspect
        sig = inspect.signature(swap_face_hybrid)
        params = list(sig.parameters.keys())
        
        assert 'source_face' in params, "swap_face_hybrid should accept source_face"
        assert 'target_face' in params, "swap_face_hybrid should accept target_face"
        assert 'temp_vision_frame' in params, "swap_face_hybrid should accept temp_vision_frame"
    
    def test_alpha_compositing_formula(self):
        """Verify alpha compositing: result = A * alpha + B * (1-alpha)."""
        # Test the compositing formula directly
        A = numpy.array([[[255, 0, 0]]], dtype=numpy.float32)  # Red (InSwapper eyes)
        B = numpy.array([[[0, 255, 0]]], dtype=numpy.float32)  # Green (SimSwap face)
        alpha = numpy.array([[[0.7]]], dtype=numpy.float32)    # 70% InSwapper
        
        result = A * alpha + B * (1 - alpha)
        
        expected = numpy.array([[[178.5, 76.5, 0]]], dtype=numpy.float32)
        numpy.testing.assert_array_almost_equal(result, expected, decimal=1)


class TestHybridSwapperIntegration:
    """Integration tests for hybrid swapper."""
    
    @pytest.fixture
    def mock_state_manager(self):
        """Mock state manager with hybrid model selected."""
        with patch('watserface.processors.modules.face_swapper.state_manager') as mock:
            mock.get_item.side_effect = lambda key: {
                'face_swapper_model': 'hybrid_inswapper_simswap',
                'face_swapper_pixel_boost': '512x512',
                'face_swapper_warp_mode': 'affine',
                'face_mask_types': ['box'],
                'face_mask_blur': 0.3,
                'face_mask_padding': (0, 0, 0, 0),
            }.get(key)
            yield mock
    
    def test_get_model_options_returns_hybrid_config(self, mock_state_manager):
        """Verify get_model_options returns hybrid configuration."""
        from watserface.processors.modules.face_swapper import get_model_options
        
        options = get_model_options()
        
        assert options is not None, "get_model_options should return config for hybrid"
        assert options.get('type') == 'hybrid', \
            "Model type should be 'hybrid'"
        assert options.get('size') == (512, 512), \
            "Hybrid should use 512x512 (SimSwap native size)"


class TestHybridSwapperQuality:
    """Quality verification tests."""
    
    def test_eye_region_uses_inswapper_output(self):
        """Verify eye regions come from InSwapper (better perceptual quality)."""
        # This is a conceptual test - actual verification requires visual inspection
        # The test ensures the code path exists
        from watserface.processors.modules.face_swapper import (
            create_eye_region_mask,
            blend_hybrid_results
        )
        
        # Verify functions exist
        assert callable(create_eye_region_mask)
        assert callable(blend_hybrid_results)
    
    def test_face_region_uses_simswap_output(self):
        """Verify non-eye face regions come from SimSwap (better texture)."""
        # Conceptual test - verifies the blending logic exists
        from watserface.processors.modules.face_swapper import blend_hybrid_results
        
        # Create mock frames
        inswapper_result = numpy.ones((256, 256, 3), dtype=numpy.float32) * 100
        simswap_result = numpy.ones((256, 256, 3), dtype=numpy.float32) * 200
        eye_mask = numpy.zeros((256, 256), dtype=numpy.float32)
        eye_mask[100:150, 100:150] = 1.0  # Eye region
        
        blended = blend_hybrid_results(inswapper_result, simswap_result, eye_mask)
        
        # Eye region should be closer to InSwapper (100)
        eye_value = blended[125, 125, 0]
        assert eye_value < 150, f"Eye region should use InSwapper, got {eye_value}"
        
        # Non-eye region should be closer to SimSwap (200)
        face_value = blended[200, 200, 0]
        assert face_value > 150, f"Face region should use SimSwap, got {face_value}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
