import numpy
import pytest
from watserface import face_masker

def test_constants_exist():
    assert hasattr(face_masker, 'IMAGENET_MEAN')
    assert hasattr(face_masker, 'IMAGENET_STD')
    assert isinstance(face_masker.IMAGENET_MEAN, numpy.ndarray)
    assert isinstance(face_masker.IMAGENET_STD, numpy.ndarray)
    assert face_masker.IMAGENET_MEAN.dtype == numpy.float32
    assert face_masker.IMAGENET_STD.dtype == numpy.float32

    # Verify values
    numpy.testing.assert_array_almost_equal(face_masker.IMAGENET_MEAN, numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32))
    numpy.testing.assert_array_almost_equal(face_masker.IMAGENET_STD, numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32))
