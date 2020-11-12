import numpy as np
from rotation_finder import get_proper_rotation
from utils import rotate_image, combine_save_patches, random_rotation_angle
from tqdm.auto import tqdm


def checkSome(only_convolutional, x_test, y_test, model, examples, options):
    to_test = []
    rotationError = 0.0
    backRotationTotal = 0.0
    for i in tqdm(range(options.samples), disable=options.serial):
        rotation = random_rotation_angle(options.step)
        rotated = rotate_image(x_test[i], rotation)

        # Original get_proper_rotation:
        properRotation = get_proper_rotation(only_convolutional, rotated, examples, i, options)

        backRotatedImage = rotate_image(rotated, properRotation)
        rotationError += (360 - (rotation + properRotation)) ** 2
        backRotationTotal += properRotation

        if options.debug:
            combine_save_patches(rotated, i + 1, 'ARot_img_' + str(i))
            combine_save_patches(backRotatedImage, i + 1, 'BRot_img_' + str(i))

        to_test.append(backRotatedImage)

    if options.debug:
        print("Rotation error: " + str(rotationError / options.samples))
        print("Average rotation calculated: " + str(backRotationTotal / options.samples))

    y_to_test = y_test[:options.samples]
    to_test = np.array(to_test, 'float32')
    verbose = 2
    if options.serial:
        verbose = 0
    return model.evaluate(np.expand_dims(to_test, -1), y_to_test, verbose=verbose)
