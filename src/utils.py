"""various utilities for modifying numpy array images etc"""
import os
import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa


def get_new_length(image):
    """get new length"""
    new_length = max(image.shape[0], image.shape[1])
    new_length = math.ceil(math.sqrt(2) * new_length)
    return new_length


def enlarge_image(image):
    """Enlarge an image, allowing it to be rotated without losing data"""
    return enlarge_images(tf.expand_dims(image, 0))[0]


def enlarge_images(images):
    # First we find the "square" part of the tensor
    start = len(images.shape) -3
    assert images.shape[start] == images.shape[start+1]
    new_length = math.ceil(math.sqrt(2) * images.shape[start])
    offset = int((new_length - images.shape[start]) / 2)
    padding = tf.constant(
        [[0, 0]] * (start)
        + [[offset, offset], [offset, offset]]
        + [[0, 0]] * (len(images.shape) - start - 2)
    )
    return tf.pad(images, padding, constant_values=0)


def random_rotation_angle(step):
    """calculate a random rotation"""
    return int(randint(0, int(360 / step)) * step)


def random_rotate_images(images, step):
    rotations = [random_rotation_angle(step) for i in range(len(images))]
    return tfa.image.rotate(images, step)

def combine_patches(patches):
    """Combine patches to a single image"""
    if not isinstance(patches, np.ndarray):
        patches = patches.numpy()
    shape = 1
    for i in patches.shape:
        shape *= i
    a = int(math.sqrt(shape))
    b = int(shape / a)
    while a * b != shape and a > 0:
        a -= 1
        b = int(shape / a)
    return np.reshape(patches, (a, b), 'F')


def combine_save_patches(patches, indx, name=''):
    """Save the combined patches to an image file"""
    combined_patches = combine_patches(patches)

    save_dir = "/tmp/P5/RotatedImg/"
    indx_dir = save_dir + str(indx) + "/"

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(indx_dir):
        os.mkdir(indx_dir)

    save_file = indx_dir + name + ".png"
    plt.imsave(save_file, combined_patches, cmap='gray')
