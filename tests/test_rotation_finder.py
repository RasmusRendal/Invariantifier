import numpy as np
import rotation_finder
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))


class RotationFinderTester(unittest.TestCase):
    def test_np_array_len(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9, 10], [11, 12, 13, 14]])
        c = np.array([[1, 2, 3], [4, 5, 6]])
        d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert rotation_finder.np_array_len(a) == 6
        assert rotation_finder.np_array_len(b) == 8
        assert rotation_finder.np_array_len(d) == 9
        assert rotation_finder.np_array_len(
            a) == rotation_finder.np_array_len(c)

    def test_cmp_np_arrays(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([[7, 8, 9], [10, 11, 12]])
        c = np.array([[1, 2, 3], [4, 5, 6]])
        d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert rotation_finder.cmp_np_arrays(a, a) is True
        assert rotation_finder.cmp_np_arrays(a, c) is True
        assert rotation_finder.cmp_np_arrays(a, b) is False
        assert rotation_finder.cmp_np_arrays(a, d) is False
        assert rotation_finder.cmp_np_arrays(d, a) is False


if __name__ == '__main__':
    unittest.main()
