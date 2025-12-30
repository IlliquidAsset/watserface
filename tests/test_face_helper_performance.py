import numpy
from collections import namedtuple
from facefusion.face_helper import predict_next_faces

# Mock Face object
Face = namedtuple('Face',
[
	'bounding_box',
	'score_set',
	'landmark_set',
	'angle',
	'embedding',
	'normed_embedding',
	'gender',
	'age',
	'race'
])

def create_dummy_face(x, y):
    bbox = numpy.array([x, y, x+50, y+50], dtype=numpy.float32)
    # create dummy data for other fields
    return Face(
        bounding_box=bbox,
        score_set={},
        landmark_set={
            '5': numpy.zeros((5, 2)),
            '5/68': numpy.zeros((5, 2)),
            '68': numpy.zeros((68, 2)),
            '68/5': numpy.zeros((68, 2))
        },
        angle=0,
        embedding=numpy.zeros(512),
        normed_embedding=numpy.zeros(512),
        gender='male',
        age=25,
        race='white'
    )

def test_predict_next_faces_logic():
    # Setup - similar faces shifted
    face1 = create_dummy_face(10, 10)
    face2 = create_dummy_face(100, 100)

    prev_faces = [face1, face2]

    # Last faces shifted slightly
    last_face1 = create_dummy_face(15, 15) # Should match face1
    last_face2 = create_dummy_face(105, 105) # Should match face2

    last_faces = [last_face1, last_face2]

    face_history = [prev_faces, last_faces]

    result = predict_next_faces(face_history)

    assert len(result) == 2
    # Verify that the new faces are updated (based on logic in predict_next_faces)
    # It creates a new Face object if match found

    assert result[0].bounding_box[0] > 10 # Should be shifted

    # Check that we didn't break functionality
    # The logic is:
    # closest_face = prev_face
    # delta_bbox = last_face.bounding_box - closest_face.bounding_box
    # new_bbox = last_face.bounding_box + delta_bbox
    # last_face (15) - closest (10) = 5
    # new_bbox = 15 + 5 = 20

    # Wait, let's trace the logic in predict_next_faces:
    # delta_bbox = last_face.bounding_box - closest_face.bounding_box
    # new_bbox = last_face.bounding_box + delta_bbox

    # So if last is 15, closest is 10. delta is 5. new is 15+5=20.

    assert abs(result[0].bounding_box[0] - 20) < 1.0
