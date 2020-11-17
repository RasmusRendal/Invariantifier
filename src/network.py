#!/usr/bin/env python3
"""module for various NN-related models and utilities"""
import tensorflow as tf
import numpy as np
from src.trainer import train_and_test
from src.utils import enlarge_images


def split_network(model, num_layers):
    """Returns the the network up to and from the specified layer"""
    if num_layers == 0:
        return None, model
    part1 = tf.keras.models.Sequential(model.layers[:num_layers])
    part2 = tf.keras.models.Sequential(model.layers[num_layers:])

    part1.compile(optimizer='adam')
    part2.compile(optimizer='adam')
    part1.build(input_shape=model.input_shape)
    part2.build(input_shape=part2.layers[0].input_shape)
    weights = model.get_weights()
    part1.set_weights(weights[:len(part1.get_weights())])
    part2.set_weights(weights[len(part1.get_weights()):])
    return part1, part2

def get_dataset(options):
    """Gets the MNIST dataset, and enlarge the images if options.enlarge is set to true"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if options.enlarge:
        x_train = enlarge_images(x_train)
        x_test = enlarge_images(x_test)
    return (x_train, y_train, x_test, y_test)


def get_model(x_test):
    """initialize convolutional model"""
    # Unless something changes, (40, 40, 1)
    input_shape = x_test[0].shape + (1,)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=(3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Conv2D(64, kernel_size=(5), strides=2, padding='same', activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Conv2D(64, kernel_size=(3), activation="relu", input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=(3), activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, kernel_size=(5), strides=2, padding='same', activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model


def image_to_model(image):
    """convert image to model"""
    return np.expand_dims(np.expand_dims(image, 0), -1)


def train_network(model, options):
    """Returns a trained network, and a cutted version of the trained network"""
    print(train_and_test(model, options))
    return model
