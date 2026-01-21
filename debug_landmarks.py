import cv2
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from watserface import state_manager, face_landmarker, vision
from watserface.types import VisionFrame

def draw_landmarks(image, landmarks, color=(0, 255, 0)):
    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), 2, color, -1)

def run_debug(target_path):
    print(f"Debug: Analyzing {target_path}")
    frame = cv2.imread(target_path)
    if frame is None:
        print("Error loading image")
        return

    # Initialize Face Detector (YOLO by default)
    from watserface import face_analyser # Fix: Use face_analyser
    state_manager.init_item('face_detector_model', 'yolo_face')
    state_manager.init_item('face_detector_size', '640x640')
    state_manager.init_item('face_detector_score', 0.5)
    state_manager.init_item('face_detector_angles', [0]) # Fix: Initialize angles
    state_manager.init_item('face_landmarker_score', 0.5) # Fix: Initialize landmarker score
    state_manager.init_item('download_providers', ['github'])
    state_manager.init_item('execution_providers', ['cpu'])
    state_manager.init_item('execution_device_id', '0')
    
    # 1. Run 2DFAN4
    print("Running 2DFAN4...")
    state_manager.init_item('face_landmarker_model', '2dfan4')
    
    # Use get_many_faces to get Face objects
    faces_2dfan4 = face_analyser.get_many_faces([frame]) 
    if not faces_2dfan4:
        print("No faces detected for 2DFAN4")
    
    frame_2dfan4 = frame.copy()
    for face in faces_2dfan4:
        # Detect landmarks
        landmarks, _, score = face_landmarker.detect_face_landmark(frame, face.bounding_box, face.angle)
        if landmarks is not None:
            draw_landmarks(frame_2dfan4, landmarks, (0, 255, 0))
            # Draw bbox
            bbox = face.bounding_box
            cv2.rectangle(frame_2dfan4, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)

    cv2.imwrite("previewtest/debug_landmarks_2dfan4.png", frame_2dfan4)

    # 2. Run MediaPipe
    print("Running MediaPipe...")
    state_manager.init_item('face_landmarker_model', 'mediapipe')
    
    faces_mp = face_analyser.get_many_faces([frame])
    
    frame_mp = frame.copy()
    for face in faces_mp:
        landmarks, landmarks_478, score = face_landmarker.detect_face_landmark(frame, face.bounding_box, face.angle)
        
        if landmarks is not None:
            # Draw 68 points (Red)
            draw_landmarks(frame_mp, landmarks, (0, 0, 255))
        
        # Draw 478 points (Yellow) if available (to check coverage)
        # Note: landmarks_478 in code might be normalized or not depending on the return type
        # The code I read said: landmarks_478 = numpy.array([ [ lm.x * w, lm.y * h, lm.z * w ] ...
        # So it should be absolute coordinates.
        if landmarks_478 is not None:
             # Just draw a few to verify
             for i, point in enumerate(landmarks_478):
                 x, y = point[:2] # Handle 2D or 3D
                 if i % 10 == 0: # Sparse
                    cv2.circle(frame_mp, (int(x), int(y)), 1, (0, 255, 255), -1)

    cv2.imwrite("previewtest/debug_landmarks_mediapipe.png", frame_mp)
    print("Done. Check previewtest/debug_landmarks_*.png")

if __name__ == "__main__":
    run_debug("/Users/kendrick/Documents/zBam_noOcclusion.png")
