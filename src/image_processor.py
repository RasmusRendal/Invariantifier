import tensorflow as tf
from src.utils import enlarge_images


@tf.function
def process_image(image, only_convolutional, options):
    """Process the image according to the options so it can be compared to training samples"""
    return process_images(
        tf.expand_dims(
            image,
            0),
        only_convolutional,
        options)[0]


@tf.function
def process_images(images, only_convolutional, options):
    if options.convlayers != 0:
        images = only_convolutional(images)
    if options.enlarge:
        images = enlarge_images(images)
    return images
