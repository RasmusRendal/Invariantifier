import tensorflow as tf
from network import image_to_model
from utils import enlarge_image, enlarge_images, combine_patches


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
        images = only_convolutional.predict(images)
    if options.combine and options.convlayers != 0:
        print(images.shape)
    if options.enlarge:
        images = enlarge_images(images)
    return images
