import os.path
import numpy as np
import tensorflow as tf
from src.utils import enlarge_images, random_rotate_images

# try:
#     tf.compat.v1.enable_eager_execution()
# except:
#     pass


def train_and_test(model, options):
    """train the neural network based on the options given"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    enlarged = ""
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)
    if options.enlarge:
        enlarged = "_enlarged"
        x_train = enlarge_images(x_train)
        x_test = enlarge_images(x_test)

    model_path = "/tmp/P5/model" + enlarged + "/P5_NN_Model.ckpt"
    model_dir = os.path.dirname(model_path)

    model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                        save_weights_only=True,
                                                        verbose=1)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    if os.path.isdir(model_dir):
        latest = tf.train.latest_checkpoint(model_dir)
        model.load_weights(latest)

        print("Loaded weights from: " + model_path)
    else:
        model.fit(x_train, y_train, epochs=5, callbacks=[model_callback])

        model.save(model_path)

        print("Saved model to: " + model_path)

    first_eval = model.evaluate(x_test, y_test, verbose=2)
    if first_eval[1] < 0.9:
        raise ValueError("Network too ineffecient (" + str(first_eval[1]) + ")")
    random_rotate_images(x_test, options.step)
    second_eval = model.evaluate(x_test, y_test, verbose=2)
    return (first_eval, second_eval)
