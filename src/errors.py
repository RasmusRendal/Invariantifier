"""module for calculating various error rates for testing correctness"""
import tensorflow as tf


def total_length(arr):
    """calc length of numpy array
       TODO: return len(arr)?"""
    length = 1
    for i in arr.shape:
        length *= i
    return length


def all_squared_errors(one, rest):
    """Gets all the squared errors between the element in one, and
    all the elements in `rest`"""
    if len(one.shape) == 2:
        one = tf.expand_dims(one, 0)
    return tf.reduce_sum(
            tf.reduce_sum(
                tf.math.squared_difference(one, rest), 1), 1)
