import tensorflow as tf
from src.utils import enlarge_images


def process_image(image, only_convolutional, options):
    """Process the image according to the options so it can be compared to training samples"""
    return process_images(
        tf.expand_dims(
            image,
            0),
        only_convolutional,
        options)[0]


def process_images(images, only_convolutional, options):
    if options.convlayers != 0:
        if images.shape[-1] != 1:
            images = tf.expand_dims(images, -1)
        images = only_convolutional.predict(images)
    if options.enlarge:
        images = enlarge_images(images)
    return images
