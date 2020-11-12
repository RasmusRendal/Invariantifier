import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import utils
import unittest


class UtilsTester(unittest.TestCase):
    def test_random_rotation_angle(self):
        for i in range(0, 360, 20):
            if i == 0:
                continue
            assert utils.random_rotation_angle(i) % i == 0


if __name__ == '__main__':
    unittest.main()
