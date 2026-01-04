
import unittest
from unittest.mock import patch, MagicMock
import platform
import watserface.common_helper

class TestCommonHelper(unittest.TestCase):
    def test_platform_checks_consistency(self):
        """Verify that platform checks return boolean and are consistent with _SYSTEM."""
        is_linux = watserface.common_helper.is_linux()
        is_macos = watserface.common_helper.is_macos()
        is_windows = watserface.common_helper.is_windows()

        self.assertIsInstance(is_linux, bool)
        self.assertIsInstance(is_macos, bool)
        self.assertIsInstance(is_windows, bool)

        # Only one should be true (or none if other OS)
        true_count = sum([is_linux, is_macos, is_windows])
        self.assertLessEqual(true_count, 1)

    def test_cached_value_correctness(self):
        """Verify the cached value matches the actual system call (at least once)."""
        current_system = platform.system().lower()
        if current_system == 'linux':
            self.assertTrue(watserface.common_helper.is_linux())
            self.assertFalse(watserface.common_helper.is_macos())
            self.assertFalse(watserface.common_helper.is_windows())
        elif current_system == 'darwin':
            self.assertFalse(watserface.common_helper.is_linux())
            self.assertTrue(watserface.common_helper.is_macos())
            self.assertFalse(watserface.common_helper.is_windows())
        elif current_system == 'windows':
            self.assertFalse(watserface.common_helper.is_linux())
            self.assertFalse(watserface.common_helper.is_macos())
            self.assertTrue(watserface.common_helper.is_windows())

if __name__ == '__main__':
    unittest.main()
