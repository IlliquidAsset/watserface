import numpy
import math
from watserface.face_helper import estimate_face_angle

def test_estimate_face_angle_correctness():
    # Helper to generate dummy 68 landmarks that produce a specific angle
    def create_landmarks_for_angle(angle_deg):
        # We need landmarks[0] and landmarks[16] to form the angle.
        # tan(theta) = dy / dx
        rad = math.radians(angle_deg)
        x1, y1 = 0, 0
        length = 100
        x2 = length * math.cos(rad)
        y2 = length * math.sin(rad)

        landmarks = numpy.zeros((68, 2))
        landmarks[0] = [x1, y1]
        landmarks[16] = [x2, y2]
        return landmarks

    # Test cases: (input_angle, expected_snapped_angle)
    # The logic is: snap to nearest 90 degrees.
    # If exactly in between (e.g., 45, 135), it snaps to the LOWER angle (based on original implementation behavior).
    # 0-45 -> 0
    # 45 -> 0
    # 46 -> 90
    # 90 -> 90
    # 134 -> 90
    # 135 -> 90
    # 136 -> 180

    test_cases = [
        (0, 0),
        (44, 0),
        (45, 0),
        (46, 90),
        (89, 90),
        (90, 90),
        (91, 90),
        (134, 90),
        (135, 90),
        (136, 180),
        (179, 180),
        (180, 180),
        (181, 180),
        (224, 180),
        (225, 180),
        (226, 270),
        (269, 270),
        (270, 270),
        (271, 270),
        (314, 270),
        (315, 270),
        (316, 0), # 360 % 360 = 0
        (359, 0),
        (360, 0),
        (1, 0)
    ]

    for input_angle, expected in test_cases:
        landmarks = create_landmarks_for_angle(input_angle)
        result = estimate_face_angle(landmarks)
        assert result == expected, f"Input {input_angle}: Expected {expected}, got {result}"
