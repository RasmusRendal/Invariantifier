from tqdm.auto import tqdm

import numpy as np
from rotation_finder import get_proper_rotation
from utils import rotate_image, combine_save_patches, random_rotation_angle


def check_some(only_convolutional, x_test, y_test, model, examples, options):
    to_test = []
    rotation_error = 0.0
    back_rotation_total = 0.0
    for i in tqdm(range(options.samples), disable=options.serial):
        rotation = random_rotation_angle(options.step)
        rotated = rotate_image(x_test[i], rotation)

        # Original get_proper_rotation:
        proper_rotation = get_proper_rotation(
            only_convolutional, rotated, examples, i, options)

        back_rotated_image = rotate_image(rotated, proper_rotation)
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
    to_test = np.array(to_test, 'float32')
    verbose = 2
    if options.serial:
        verbose = 0
    return model.evaluate(np.expand_dims(to_test, -1),
                          y_to_test, verbose=verbose)
