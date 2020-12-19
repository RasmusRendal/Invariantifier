import unittest
import sys
import os

import numpy as np
import tensorflow as tf

import src.network as network


class NetworkTests(unittest.TestCase):
    def test_constraint(self):
        equi = network.RotationEquivariant(90)
        arr = np.array([[4, 0], [0, 0]])
        expected = np.ones((2,2))
        arr = tf.cast(arr, tf.float32)
        expected = tf.cast(expected, tf.float32)
        assert (expected == equi(arr)).numpy().all()


if __name__ == '__main__':
    unittest.main()
