import sys
from unittest.mock import MagicMock

# Mock cv2 before importing watserface modules
cv2_mock = MagicMock()
cv2_mock.CAP_PROP_FRAME_COUNT = 7
cv2_mock.CAP_PROP_POS_FRAMES = 1
cv2_mock.CAP_PROP_FPS = 5
cv2_mock.typing = MagicMock()
cv2_mock.typing.Size = tuple

sys.modules['cv2'] = cv2_mock
sys.modules['cv2.typing'] = cv2_mock.typing

# Import modules to patch
import watserface.vision
import watserface.video_manager

# Setup the mock
mock_capture = MagicMock()
mock_capture.isOpened.return_value = True
mock_capture.get.return_value = 100.0 # Return 100 frames
mock_capture.read.return_value = (True, "frame_data")

# Patch get_video_capture in vision module directly to return our mock
watserface.vision.get_video_capture = lambda x: mock_capture

# Patch is_video in watserface.vision
watserface.vision.is_video = lambda x: True

print("Running reproduction script...")

# Reset mock counters
mock_capture.get.reset_mock()

# Call read_video_frame multiple times
from watserface.vision import read_video_frame
for _ in range(10):
    read_video_frame('test.mp4', 0)

# Check how many times get was called
call_count = mock_capture.get.call_count
print(f"cv2.VideoCapture.get called {call_count} times.")

if call_count > 1:
    print("Performance Issue Detected: Frame count is retrieved repeatedly.")
else:
    print("Optimization Working: Frame count is cached.")
