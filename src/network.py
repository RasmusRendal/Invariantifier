#!/usr/bin/env python3
"""module for various NN-related models and utilities"""
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from src.trainer import train_and_test
from src.utils import enlarge_images
import math


class RotationEquivariant(tf.keras.constraints.Constraint):
    def __init__(self, step):
        self.step = step

    def __call__(self, w):
        """Takes an array w, and returns w, where rot(w, 90) = w)"""
        if self.step == -1:
            return w
        out = tf.zeros(w.shape)
        for i in range(0, 360, self.step):
            out = out + tfa.image.rotate(w, math.radians(i))
        out = out / (360/self.step)
        return out


def get_last_conv_layer(model):
    """Returns the index of the last conv layer"""
    last_conv = -1
    for index, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv = index
    if last_conv == -1:
        raise ValueError("No convolution layers found")
    return last_conv

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
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    if options.enlarge:
        x_train = enlarge_images(x_train)
        x_test = enlarge_images(x_test)
    return (x_train, y_train, x_test, y_test)


def get_model(x_test, options):
    """initialize convolutional model"""
    # Unless something changes, (40, 40, 1)
    input_shape = x_test[0].shape

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu", kernel_constraint=RotationEquivariant(options.model_step), input_shape=input_shape),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(16, kernel_size=3, kernel_constraint=RotationEquivariant(options.model_step)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(4, kernel_size=3, strides=2, kernel_constraint=RotationEquivariant(options.model_step)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.summary()
    return model


def train_network(model, options):
    """Returns a trained network, and a cutted version of the trained network"""
    print(train_and_test(model, options))
    return model
