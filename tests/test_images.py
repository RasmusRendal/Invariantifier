import unittest
import numpy as np
from src.utils import enlarge_image


class TestRotations(unittest.TestCase):
    def test_enlarge_image(self):
        notEnlarged = np.zeros((3, 3, 1))
        notEnlarged[0][0][0] = 1
        notEnlarged[2][2][0] = 1

        expected = np.zeros((5, 5, 1))
        expected[1][1][0] = 1
        expected[3][3][0] = 1

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
