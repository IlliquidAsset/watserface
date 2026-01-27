"""
Test mouth interior detection for face masking.

The mouth interior mask should capture the actual mouth OPENING (cavity),
not just the lip line. When mouth is open (e.g., eating), the opening
should be 25-35px, not 8.2px (which is just the lip edge).

Root cause of bug: Inner lip landmarks (60-67) are placed at the lip EDGE
by 2dfan4, not at the actual mouth cavity opening.

Solution: Use outer lip landmarks (48-59) and expand inward to approximate
the mouth interior opening.
"""
import numpy as np
import pytest
import cv2


class TestMouthInteriorMask:
    """Test suite for mouth interior mask generation."""
    
    @pytest.fixture
    def sample_landmarks_68(self):
        """
        Create sample 68-point landmarks simulating an open mouth.
        
        Landmark indices for mouth:
        - Outer lip: 48-59 (12 points forming outer lip contour)
        - Inner lip: 60-67 (8 points forming inner lip contour)
        
        In 2dfan4, inner lip points (60-67) are placed at the lip EDGE,
        not at the actual mouth opening. This is the bug we're fixing.
        """
        # Create a 68-point landmark array (x, y coordinates)
        landmarks = np.zeros((68, 2), dtype=np.float32)
        
        center_x, center_y = 128, 140
        
        outer_lip_width = 45
        outer_lip_height = 38
        
        for i, idx in enumerate(range(48, 55)):
            angle = np.pi - (i / 6) * np.pi
            landmarks[idx] = [
                center_x + outer_lip_width * np.cos(angle),
                center_y - outer_lip_height * 0.5 * np.sin(angle)
            ]
        
        for i, idx in enumerate(range(55, 60)):
            angle = (i / 4) * np.pi
            landmarks[idx] = [
                center_x + outer_lip_width * np.cos(angle),
                center_y + outer_lip_height * 0.5 * np.sin(angle)
            ]
        
        inner_lip_width = 35
        inner_lip_height = 4
        
        for i, idx in enumerate(range(60, 65)):
            angle = np.pi - (i / 4) * np.pi
            landmarks[idx] = [
                center_x + inner_lip_width * np.cos(angle),
                center_y - inner_lip_height * 0.5 * np.sin(angle)
            ]
        
        for i, idx in enumerate(range(65, 68)):
            angle = (i / 2) * np.pi
            landmarks[idx] = [
                center_x + inner_lip_width * np.cos(angle),
                center_y + inner_lip_height * 0.5 * np.sin(angle)
            ]
        
        return landmarks
    
    @pytest.fixture
    def crop_vision_frame(self):
        """Create a sample 256x256 crop frame."""
        return np.zeros((256, 256, 3), dtype=np.uint8)
    
    def test_mouth_interior_mask_size_minimum(self, sample_landmarks_68, crop_vision_frame):
        """
        Test that mouth interior mask captures actual opening, not just lip line.
        
        Expected: Mouth opening height should be 25-35px (actual cavity)
        Bug baseline: Inner lip landmarks give only 8.2px (lip edge)
        """
        from watserface.face_masker import create_mouth_interior_mask
        
        mask = create_mouth_interior_mask(crop_vision_frame, sample_landmarks_68)
        
        # Find the bounding box of the mask
        mask_binary = (mask > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        assert len(contours) > 0, "Mouth interior mask should have at least one contour"
        
        # Get bounding rect of largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # The mouth opening height should be at least 25px (not 8.2px bug)
        assert h >= 25, f"Mouth interior height {h}px is too small (expected >= 25px, bug baseline was 8.2px)"
        assert h <= 45, f"Mouth interior height {h}px is too large (expected <= 45px)"
    
    def test_mouth_interior_mask_uses_outer_lip_expansion(self, sample_landmarks_68, crop_vision_frame):
        """
        Test that mouth interior mask is derived from outer lip with inward expansion,
        NOT from inner lip landmarks (which are buggy).
        """
        from watserface.face_masker import create_mouth_interior_mask
        
        mask = create_mouth_interior_mask(crop_vision_frame, sample_landmarks_68)
        
        # The mask should be smaller than outer lip but larger than inner lip
        # Calculate expected areas
        outer_lip_points = sample_landmarks_68[48:60].astype(np.int32)
        inner_lip_points = sample_landmarks_68[60:68].astype(np.int32)
        
        outer_hull = cv2.convexHull(outer_lip_points)
        inner_hull = cv2.convexHull(inner_lip_points)
        
        outer_area = cv2.contourArea(outer_hull)
        inner_area = cv2.contourArea(inner_hull)
        
        # Mask area should be between inner and outer (closer to outer * 0.6-0.8)
        mask_binary = (mask > 0.5).astype(np.uint8)
        mask_area = np.sum(mask_binary)
        
        # Mouth interior should be significantly larger than buggy inner lip area
        assert mask_area > inner_area * 2, (
            f"Mouth interior area {mask_area} should be > 2x inner lip area {inner_area}"
        )
    
    def test_mouth_interior_mask_shape(self, sample_landmarks_68, crop_vision_frame):
        """Test that the mask has correct shape matching input frame."""
        from watserface.face_masker import create_mouth_interior_mask
        
        mask = create_mouth_interior_mask(crop_vision_frame, sample_landmarks_68)
        
        assert mask.shape == crop_vision_frame.shape[:2][::-1], (
            f"Mask shape {mask.shape} should match frame shape {crop_vision_frame.shape[:2][::-1]}"
        )
        assert mask.dtype == np.float32, f"Mask dtype should be float32, got {mask.dtype}"
    
    def test_mouth_interior_mask_values_normalized(self, sample_landmarks_68, crop_vision_frame):
        """Test that mask values are properly normalized to [0, 1]."""
        from watserface.face_masker import create_mouth_interior_mask
        
        mask = create_mouth_interior_mask(crop_vision_frame, sample_landmarks_68)
        
        assert mask.min() >= 0.0, f"Mask min {mask.min()} should be >= 0"
        assert mask.max() <= 1.0, f"Mask max {mask.max()} should be <= 1"


class TestMouthInteriorMaskIntegration:
    """Integration tests for mouth interior mask with real-world scenarios."""
    
    def test_mouth_interior_preserves_food_object(self):
        """
        Test that mouth interior mask is large enough to preserve objects
        inside the mouth (e.g., corndog in eating scene).
        
        This is the key use case: when swapping faces, we need to preserve
        what's inside the mouth, not just the lip line.
        """
        # This test requires the actual video frame - skip if not available
        pytest.skip("Integration test requires actual video frame from zBambola.mp4")
