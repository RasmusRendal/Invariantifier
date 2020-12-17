import math

import tensorflow as tf
import tensorflow_addons as tfa
from tqdm.auto import tqdm

from src.rotation_finder import get_proper_rotation
from src.utils import random_rotation_angle, combine_save_patches

#pylint: disable=too-many-arguments,too-many-locals
def check_some(only_convolutional, x_test, y_test, model, examples, options):
    to_test = []
    rotation_error = 0.0
    back_rotation_total = 0.0
    for i in tqdm(range(options.samples), disable=options.serial):
        rotation = random_rotation_angle(options.step)
        rotated = tfa.image.rotate(x_test[i], math.radians(rotation))

        # Original get_proper_rotation:
        proper_rotation = int(get_proper_rotation(
            only_convolutional, rotated, examples, options))

        back_rotated_image = tfa.image.rotate(rotated, math.radians(proper_rotation))
        rotation_error += (360 - (rotation + proper_rotation)) ** 2
        back_rotation_total += proper_rotation

        if options.debug:
            combine_save_patches(rotated, i + 1, 'ARot_img_' + str(i))
            combine_save_patches(
                back_rotated_image,
                i + 1,
                'BRot_img_' + str(i))

        to_test.append(back_rotated_image)

    if options.debug:
        print("Rotation error: " + str(rotation_error / options.samples))
        print("Average rotation calculated: " +
              str(back_rotation_total / options.samples))

    y_to_test = y_test[:options.samples]
    to_test = tf.cast(to_test, tf.float32)
    verbose = 2
    if options.serial:
        verbose = 0
    if options.accperclass:
        return eval_model(model, tf.expand_dims(to_test, -1), y_to_test, verbose)

    return model.evaluate(tf.expand_dims(to_test, -1),
                              y_to_test, verbose=verbose)

def eval_model(model, to_test, y_to_test, verbose):
    res = [0]*11
    temp_per_class = [0]*10
    temp_total = 0
    to_compare = model.predict(to_test, verbose=verbose)
    to_compare = tf.math.argmax(to_compare, 1).numpy()

    for i, _ in enumerate(to_compare):
        j = to_compare[i]
        k = y_to_test[i]
        if j == k:
            temp_total = temp_total + 1
            res[j] = res[j] + 1
        temp_per_class[k] = temp_per_class[k] + 1

    for i in range(10):
        res[i] = res[i] / temp_per_class[i]

    res[10] = temp_total / len(to_compare)

    return res
