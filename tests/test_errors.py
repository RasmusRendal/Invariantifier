import tensorflow as tf
import numpy as np
import unittest
import errors
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))


class ErrorsTesting(unittest.TestCase):
    def test_total_length(self):
        a = np.array([1, 2, 3])
        d = np.array([[1, 2, 3], [4, 5, 6]])
        assert errors.total_length(d) == 6
        assert errors.total_length(a) == 3

    def test_gpu_squared_error(self):
        dims = (2, 2)
        firstImage = np.zeros(dims)
        secondImage = np.zeros(dims)
        firstImage[0][0] = 1
        firstImage[0][1] = 1
        secondImage[0][0] = 1
        thirdImage = np.zeros(dims)
        second = tf.constant(
            [secondImage, thirdImage, secondImage], dtype=np.float32)
        firstTF = tf.constant(firstImage, dtype=np.float32)
        for _ in range(10):
            all_error = errors.all_squared_errors(firstTF, second)
            self.assertEqual((3,), all_error.shape)
            self.assertEqual(1, all_error[0])
            self.assertEqual(2, all_error[1])
            self.assertEqual(1, all_error[2])


if __name__ == '__main__':
    unittest.main()
