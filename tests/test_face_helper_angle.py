
import pytest
import numpy
from watserface.face_helper import estimate_face_angle

def test_estimate_face_angle():
	# Test case 1: 0 degrees
	# Points 0 and 16 horizontally aligned
	lm = numpy.zeros((68, 2))
	lm[0] = [0, 0]
	lm[16] = [100, 0]
	angle = estimate_face_angle(lm)
	assert angle == 0

	# Test case 2: 90 degrees
	# Points 0 and 16 vertically aligned (0 at bottom, 16 at top?)
	# dy = 100, dx = 0. atan2(100, 0) = 90 deg.
	lm = numpy.zeros((68, 2))
	lm[0] = [0, 0]
	lm[16] = [0, 100]
	angle = estimate_face_angle(lm)
	assert angle == 90

	# Test case 3: 45 degrees
	# atan2(100, 100) = 45 deg.
	# Optimized logic: ceil(45/90 - 0.5) = ceil(0) = 0.
	lm = numpy.zeros((68, 2))
	lm[0] = [0, 0]
	lm[16] = [100, 100]
	angle = estimate_face_angle(lm)
	assert angle == 0

	# Test case 4: 135 degrees
	# atan2(100, -100) = 135 deg.
	# Optimized logic: ceil(135/90 - 0.5) = ceil(1.5-0.5) = 1. 1*90 = 90.
	lm = numpy.zeros((68, 2))
	lm[0] = [0, 0]
	lm[16] = [-100, 100]
	angle = estimate_face_angle(lm)
	assert angle == 90

	# Test case 5: 180 degrees
	# atan2(0, -100) = 180 deg.
	lm = numpy.zeros((68, 2))
	lm[0] = [0, 0]
	lm[16] = [-100, 0]
	angle = estimate_face_angle(lm)
	assert angle == 180

	# Test case 6: 270 degrees
	# atan2(-100, 0) = -90 = 270 deg.
	lm = numpy.zeros((68, 2))
	lm[0] = [0, 0]
	lm[16] = [0, -100]
	angle = estimate_face_angle(lm)
	assert angle == 270
