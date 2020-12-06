"""The main algorithm is in this file. The entrypoint will be
get_proper_rotation"""
import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from src.image_processor import process_image, process_images
from src.utils import combine_save_patches 


def np_array_len(array):
    length = 1
    for i in array.shape:
        length *= i
    return length


def cmp_np_arrays(arr1, arr2):
    len1 = np_array_len(arr1)
    len2 = np_array_len(arr2)
    if len1 != len2:
        return False
    arr1 = np.reshape(arr1, (len1,))
    arr2 = np.reshape(arr2, (len1,))
    for i in range(len1):
        if arr1[i] != arr2[i]:
            return False
    return True


def get_proper_rotation(only_convolutional,
                        image,
                        training_samples,
                        imgindx,
                        options):
    """Get the rotation to apply to a network to make it recognizable"""
    rotations = None
    if options.rotate_first:
        rotations = get_rotations(image, options)
        rotations = process_images(rotations, only_convolutional, options)
    else:
        image = process_image(image, only_convolutional, options)
        rotations = get_rotations(image, options)

    if options.debug:
        for index, rotation in enumerate(rotations):
            combine_save_patches(
                rotation,
                imgindx,
                'ConvRots_img_' +
                str(index))
        for (index1, rotation1), (index2, rotation2) in zip(
                enumerate(rotations), enumerate(rotations)):
            same = cmp_np_arrays(rotation1, rotation2)
            if same:
                print(
                    "Index " +
                    str(index1) +
                    " and index " +
                    str(index2) +
                    " are identical")

    return get_best_rotation(
        training_samples,
        rotations,
        options)


def get_rotations(image, options):
    """Create all the possible rotations of an image, according to the options"""
    rotations = None
    if options.combine:
        rotations_to_try = int(360 / options.step)
        rotations = tf.tile(tf.expand_dims(tf.expand_dims(
            image, -1), 0), [rotations_to_try, 1, 1, 1])
        rotations = tfa.image.rotate(
            rotations, [i * math.pi / 180 for i in range(0, 360, options.step)])
        rotations = tf.squeeze(rotations)
    else:
        if options.convlayers == 0:
            rotations = [tfa.image.rotate(image, math.radians(i))
                         for i in range(0, 360, options.step)]
        else:
            rotation_angles = [
                math.radians(i) for i in range(
                    0, 360, options.step)]
            rotated = []
            for angle in rotation_angles:
                rotated.append(tfa.image.rotate(image, angle))
            rotations = tf.stack(rotated)
            assert rotations.shape == (len(rotation_angles),) + image.shape
    return tf.cast(rotations, tf.float32)


def get_best_rotation(training_samples, rotations, options):
    """This function is a bit of a complicated matrix operation.
    Given a tensor training_samples of dimension m * i, and tensor
    rotations n * i, where m and n are scalars, and i are any subsequent
    sets of dimensions, it will build a tensor m*n where A_m_n is the error
    between m and n. It will then find the index i,j with the lowest error,
    returning j (The rotation)
    """
    if rotations.shape[1:] != training_samples.shape[1:]:
        raise ValueError("Incompatible shapes: {0} and {1}".format(rotations.shape, training_samples.shape))

    # Expand the dimensions of the vectors to be (1, n) + i
    # and (n, 1) + 1. This makes the arrays compatible using
    # broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
    rotations = tf.expand_dims(rotations, 0)
    training_samples = tf.expand_dims(training_samples, 1)

    # Squared difference per pixel
    error = tf.math.squared_difference(rotations, training_samples)

    # Reduce all axes, besides the two first, by summing
    reduce_axes = list(range(2, len(error.shape)))
    error = tf.reduce_sum(error, axis=reduce_axes)

    # We don't care which training_sample matches the best, so take the min
    error = tf.reduce_min(error, axis=0)

    # Find the minimum rotation, return it
    amin = tf.argmin(error)
    return int(amin.numpy()) * options.step
