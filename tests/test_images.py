import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import numpy as np
from utils import rotate_image, rotate_images, enlarge_image
import unittest


class TestRotations(unittest.TestCase):
    def test_rotation(self):
        notRotated = np.zeros((3, 3))
        expected = np.zeros((3, 3))
        notRotated[0][0] = 1
        notRotated[1][1] = 1
        expected[1][1] = 1
        expected[2][2] = 1
        actual = rotate_image(notRotated, 180)
        actual = np.reshape(actual, (9,))
        expected = np.reshape(expected, (9,))
        for i in range(len(actual)):
            self.assertEqual(actual[i], expected[i])

    def test_multiple_rotations(self):
        notRotated = np.zeros((3, 3))
        expected = np.zeros((3, 3))
        notRotated[0][0] = 1
        notRotated[1][1] = 1
        expected[1][1] = 1
        expected[2][2] = 1
        notRotated = np.expand_dims(notRotated, 0)
        actual = rotate_images(notRotated, 180)[0]
        actual = np.reshape(actual, (9,))
        expected = np.reshape(expected, (9,))
        for i in range(len(actual)):
            self.assertEqual(actual[i], expected[i])

    def test_enlarge_image(self):
        notEnlarged = np.zeros((3, 3))
        notEnlarged[0][0] = 1
        notEnlarged[2][2] = 1

        expected = np.zeros((5, 5))
        expected[1][1] = 1
        expected[3][3] = 1

        actual = enlarge_image(notEnlarged)

        actual = np.reshape(actual, (25,))
        expected_reshaped = np.reshape(expected, (25,))
        for x, y in zip(actual, expected_reshaped):
            self.assertEqual(x, y)

        nestedNotEnlarged = np.array([[notEnlarged]])
        nestedExpected = np.array([[expected]])
        nestedActual = enlarge_image(nestedNotEnlarged)
        self.assertEqual(nestedActual.shape, nestedExpected.shape)
        for list1, list2 in zip(nestedActual[0][0], nestedExpected[0][0]):
            for x, y in zip(list1, list2):
                self.assertEqual(x, y)


if __name__ == '__main__':
    unittest.main()
