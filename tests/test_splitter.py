import math
import unittest
import tensorflow as tf
import tensorflow_addons as tfa

from src.network import train_network, get_dataset, get_model, split_network
from src.options import Options


class TestSplitter(unittest.TestCase):
    """Test the network splitting function"""
    def test_split(self):
        options = Options()
        _, _, x_test, _ = get_dataset(options)
        model = train_network(get_model(x_test, options), options)
        part1, part2 = split_network(model, 3)
        model_output = model(x_test[0:5])
        split_output = part2(part1(x_test[0:5]))

        equality = tf.math.reduce_all(tf.equal(model_output, split_output))
        self.assertEqual(equality, True)


    def test_compose_rotation(self):
        """Test whether rotation -> convolution == convolution -> rotation
        It isn't. This has implications for our project.
        This is not as much a test as a demonstration of fact"""
        options = Options()
        _, _, x_test, _ = get_dataset(options)
        model = train_network(get_model(x_test, options), options)
        part1, _ = split_network(model, 8)
        image = tf.expand_dims(x_test[0], 0)
        rotated_image = tfa.image.rotate(image, math.pi)
        out = tfa.image.rotate(part1(image), math.pi)
        rotated_out = part1(rotated_image)
        equality = tf.math.reduce_all(tf.equal(tfa.image.rotate(out, math.pi), rotated_out))
        self.assertEqual(equality, False)


if __name__ == '__main__':
    unittest.main()
