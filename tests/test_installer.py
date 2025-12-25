from unittest.mock import MagicMock
import os


def test_library_paths_deduplication_inefficiency():
    # Setup
    library_paths = ['/path/a', '/path/b', '/path/a', '/path/c', '/path/b']

    # Mock os.path.exists
    original_exists = os.path.exists
    mock_exists = MagicMock(side_effect=lambda x: True)
    os.path.exists = mock_exists

    try:
        # Optimized logic
        # library_paths = [ library_path for library_path in dict.fromkeys(library_paths) if os.path.exists(library_path) ]
        result = [ library_path for library_path in dict.fromkeys(library_paths) if os.path.exists(library_path) ]

        # Verify result content (should be unique)
        assert result == ['/path/a', '/path/b', '/path/c']

        # Verify efficiency: called only for unique items (3 times instead of 5)
        assert mock_exists.call_count == 3

    finally:
        # Restore
        os.path.exists = original_exists
