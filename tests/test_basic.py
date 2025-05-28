"""
Basic test file to ensure the test discovery works.
"""

import unittest


class TestBasic(unittest.TestCase):
    """Basic test cases."""

    def test_always_passes(self):
        """A test that always passes."""
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
