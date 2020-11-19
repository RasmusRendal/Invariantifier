from random import randint
from tqdm.auto import tqdm
import numpy as np
import tensorflow as tf


from src.image_processor import process_images
from src.errors import all_squared_errors
from src.utils import combine_save_patches


def get_all_distances(cached, current_set, rest):
    distances = cached
    for i in range(len(cached), len(current_set)):
        # This produces a 60k length array, of the distance for
        # each element of the training set to the current set element
        distances.append(all_squared_errors(current_set[i], rest))
    return distances


def sum_all_distances(all_distances):
    return tf.reduce_min(all_distances, 0)


def get_representative_patches(x_train, examples, options):
    currentset = [x_train[0]]
    distances = []
    added_indexes = []
    for _ in tqdm(range(1, examples), disable=options.serial):
        distances = get_all_distances(distances, currentset, x_train)
        sum_distances = sum_all_distances(distances)
        maxindex = int(tf.argmax(sum_distances).numpy())
        assert maxindex not in added_indexes
        added_indexes.append(maxindex)
        currentset.append(x_train[maxindex])
    return currentset


def get_random_training_samples(x_train, examples, options): # pylint: disable=unused-argument
    indexes = [randint(0, len(x_train) - 1) for i in range(examples)]
    # Hack for indexing with an array
    if isinstance(x_train, np.ndarray):
        return x_train[indexes]
    return x_train.numpy()[indexes]


def get_seperated_representatives(function, x_train, y_train, options):
    """This function runs `function` ten times, once for each digit.
    It then combines the output of `function` for each digit, giving
    an even distribution amongst classes in the representative set"""
    training_samples = []
    # Hack for indexing with an array
    if not isinstance(x_train, np.ndarray):
        x_train = x_train.numpy()
    for i in range(10):
        idx = tf.squeeze(tf.where(y_train == i))
        training_samples.append(
            function(
                x_train[idx], int(
                    options.examples / 10), options))
    return tf.concat(training_samples, 0)


def get_training_samples(only_convolutional, x_train, y_train, options):
    training_samples = None
    if options.representatives:
        training_samples = get_seperated_representatives(
            get_representative_patches, x_train, y_train, options)
    else:
        training_samples = get_seperated_representatives(get_random_training_samples, x_train, y_train, options)
    if len(training_samples.shape) != 3:
        raise ValueError(
            "Invalid shape returned by patch finder: " + str(training_samples.shape))

    training_samples = process_images(
        training_samples, only_convolutional, options)

    if options.debug:
        for index, item in enumerate(training_samples):
            combine_save_patches(item, -1, 'Test_patches_' + str(index))
    return tf.cast(training_samples, tf.float32)
