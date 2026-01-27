"""
Test temporal stabilization for face bounding boxes and landmarks.

Target: Face size variance <5% frame-to-frame (vs current 22-30%)
Based on measurements from test_quality/ANALYSIS_FINDINGS.md
"""
import numpy
import pytest

from watserface import face_detector, face_landmarker, state_manager
from watserface.face_analyser import get_many_faces
from watserface.face_helper import (
    TemporalBBoxStabilizer,
    TemporalLandmarkStabilizer,
)
from watserface.types import BoundingBox, FaceLandmark5, FaceLandmark68


@pytest.fixture(scope='module', autouse=True)
def before_all() -> None:
    """Initialize state for temporal stabilization tests."""
    state_manager.init_item('execution_device_id', '0')
    state_manager.init_item('execution_providers', ['cpu'])
    state_manager.init_item('face_detector_model', 'yolo_face')
    state_manager.init_item('face_detector_size', '640x640')
    state_manager.init_item('face_detector_score', 0.5)
    state_manager.init_item('face_landmarker_model', '2dfan4')
    state_manager.init_item('face_landmarker_score', 0.5)


class TestTemporalBBoxStabilizer:
    """Test EMA-based bounding box stabilization."""

    def test_stabilizer_exists(self) -> None:
        """Verify TemporalBBoxStabilizer class exists and can be instantiated."""
        stabilizer = TemporalBBoxStabilizer(alpha=0.3)
        assert stabilizer is not None

    def test_stabilizer_reduces_variance(self) -> None:
        """
        Test that EMA stabilizer reduces bbox variance.
        
        Simulates the jitter pattern from ANALYSIS_FINDINGS.md:
        - Bounding box area variance: 22.8% (95,557 to 120,053 px²)
        - Largest single-frame jump: 9,326 px² (frame 22→23)
        """
        numpy.random.seed(42)
        base_bbox = numpy.array([100.0, 100.0, 400.0, 400.0])
        
        raw_bboxes = []
        for i in range(30):
            scale_jitter = numpy.random.uniform(0.85, 1.25)
            center = (base_bbox[:2] + base_bbox[2:]) / 2
            half_size = (base_bbox[2:] - base_bbox[:2]) / 2 * scale_jitter
            pos_jitter = numpy.random.uniform(-10, 10, 2)
            jittered = numpy.array([
                center[0] - half_size[0] + pos_jitter[0],
                center[1] - half_size[1] + pos_jitter[1],
                center[0] + half_size[0] + pos_jitter[0],
                center[1] + half_size[1] + pos_jitter[1]
            ])
            raw_bboxes.append(jittered)
        
        raw_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in raw_bboxes]
        raw_variance = numpy.std(raw_areas) / numpy.mean(raw_areas) * 100
        
        stabilizer = TemporalBBoxStabilizer(alpha=0.2)
        smoothed_bboxes = []
        for bbox in raw_bboxes:
            smoothed = stabilizer.update(bbox)
            smoothed_bboxes.append(smoothed)
        
        # Calculate smoothed variance
        smoothed_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in smoothed_bboxes]
        smoothed_variance = numpy.std(smoothed_areas) / numpy.mean(smoothed_areas) * 100
        
        # Assert variance is reduced significantly
        assert smoothed_variance < raw_variance, f"Smoothed variance ({smoothed_variance:.1f}%) should be less than raw ({raw_variance:.1f}%)"
        # Target: <5% variance
        assert smoothed_variance < 5.0, f"Smoothed variance ({smoothed_variance:.1f}%) should be <5%"

    def test_stabilizer_handles_first_frame(self) -> None:
        """First frame should pass through unchanged (no history)."""
        stabilizer = TemporalBBoxStabilizer(alpha=0.2)
        first_bbox = numpy.array([100.0, 100.0, 400.0, 400.0])
        result = stabilizer.update(first_bbox)
        numpy.testing.assert_array_almost_equal(result, first_bbox)

    def test_stabilizer_reset(self) -> None:
        """Test that reset clears history."""
        stabilizer = TemporalBBoxStabilizer(alpha=0.2)
        stabilizer.update(numpy.array([100.0, 100.0, 400.0, 400.0]))
        stabilizer.update(numpy.array([110.0, 110.0, 410.0, 410.0]))
        stabilizer.reset()
        
        # After reset, next frame should pass through unchanged
        new_bbox = numpy.array([200.0, 200.0, 500.0, 500.0])
        result = stabilizer.update(new_bbox)
        numpy.testing.assert_array_almost_equal(result, new_bbox)


class TestTemporalLandmarkStabilizer:
    """Test Savitzky-Golay filter for landmark stabilization."""

    def test_stabilizer_exists(self) -> None:
        """Verify TemporalLandmarkStabilizer class exists and can be instantiated."""
        stabilizer = TemporalLandmarkStabilizer(window_size=5, poly_order=2)
        assert stabilizer is not None

    def test_stabilizer_reduces_landmark_jitter(self) -> None:
        """
        Test that Savitzky-Golay filter reduces landmark jitter.
        
        Uses 5-frame window with polynomial order 2 (as specified in task).
        """
        numpy.random.seed(42)
        
        # Generate 30 frames of jittery 5-point landmarks
        base_landmarks = numpy.array([
            [150.0, 200.0],  # left eye
            [250.0, 200.0],  # right eye
            [200.0, 250.0],  # nose
            [160.0, 300.0],  # left mouth
            [240.0, 300.0],  # right mouth
        ])
        
        raw_landmarks_sequence = []
        for i in range(30):
            # Add random jitter: ±5 pixels per landmark
            jitter = numpy.random.uniform(-5, 5, base_landmarks.shape)
            jittered = base_landmarks + jitter
            raw_landmarks_sequence.append(jittered)
        
        # Calculate raw frame-to-frame movement
        raw_movements = []
        for i in range(1, len(raw_landmarks_sequence)):
            diff = numpy.linalg.norm(raw_landmarks_sequence[i] - raw_landmarks_sequence[i-1], axis=1)
            raw_movements.append(diff.mean())
        raw_avg_movement = numpy.mean(raw_movements)
        
        # Apply stabilization
        stabilizer = TemporalLandmarkStabilizer(window_size=5, poly_order=2)
        smoothed_landmarks_sequence = []
        for landmarks in raw_landmarks_sequence:
            smoothed = stabilizer.update(landmarks)
            smoothed_landmarks_sequence.append(smoothed)
        
        # Calculate smoothed frame-to-frame movement
        smoothed_movements = []
        for i in range(1, len(smoothed_landmarks_sequence)):
            diff = numpy.linalg.norm(smoothed_landmarks_sequence[i] - smoothed_landmarks_sequence[i-1], axis=1)
            smoothed_movements.append(diff.mean())
        smoothed_avg_movement = numpy.mean(smoothed_movements)
        
        # Assert movement is reduced
        assert smoothed_avg_movement < raw_avg_movement, \
            f"Smoothed movement ({smoothed_avg_movement:.2f}px) should be less than raw ({raw_avg_movement:.2f}px)"

    def test_stabilizer_handles_68_landmarks(self) -> None:
        """Test that stabilizer works with 68-point landmarks."""
        stabilizer = TemporalLandmarkStabilizer(window_size=5, poly_order=2)
        
        # Generate 68-point landmarks
        landmarks_68 = numpy.random.uniform(100, 400, (68, 2))
        
        # Should not raise
        result = stabilizer.update(landmarks_68)
        assert result.shape == (68, 2)

    def test_stabilizer_reset(self) -> None:
        """Test that reset clears history."""
        stabilizer = TemporalLandmarkStabilizer(window_size=5, poly_order=2)
        
        # Add some history
        for _ in range(5):
            stabilizer.update(numpy.random.uniform(100, 400, (5, 2)))
        
        stabilizer.reset()
        
        # After reset, buffer should be empty
        assert len(stabilizer._buffer) == 0


class TestIntegration:
    """Integration tests for temporal stabilization in face detection pipeline."""

    def test_face_size_variance_under_5_percent(self) -> None:
        """
        End-to-end test: face size variance should be <5% after stabilization.
        
        This is the key acceptance criterion from the task.
        """
        numpy.random.seed(42)
        
        base_width = 320.0
        base_height = 340.0
        base_x = 200.0
        base_y = 150.0
        
        raw_bboxes = []
        for i in range(30):
            scale_jitter = numpy.random.uniform(0.80, 1.30)
            if i == 22:
                scale_jitter = 1.35
            elif i in [12, 26]:
                scale_jitter = numpy.random.uniform(1.15, 1.30)
            
            x_jitter = numpy.random.uniform(-10, 10)
            y_jitter = numpy.random.uniform(-10, 10)
            
            bbox = numpy.array([
                base_x + x_jitter,
                base_y + y_jitter,
                base_x + base_width * scale_jitter + x_jitter,
                base_y + base_height * scale_jitter + y_jitter
            ])
            raw_bboxes.append(bbox)
        
        raw_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in raw_bboxes]
        raw_variance = numpy.std(raw_areas) / numpy.mean(raw_areas) * 100
        
        assert raw_variance > 10, f"Test setup error: raw variance ({raw_variance:.1f}%) should be >10%"
        
        stabilizer = TemporalBBoxStabilizer(alpha=0.15)
        smoothed_bboxes = []
        for bbox in raw_bboxes:
            smoothed = stabilizer.update(bbox)
            smoothed_bboxes.append(smoothed)
        
        smoothed_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in smoothed_bboxes]
        smoothed_variance = numpy.std(smoothed_areas) / numpy.mean(smoothed_areas) * 100
        
        assert smoothed_variance < 5.0, \
            f"Face size variance ({smoothed_variance:.1f}%) exceeds 5% target. " \
            f"Raw variance was {raw_variance:.1f}%"
