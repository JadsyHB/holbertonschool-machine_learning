#!/usr/bin/env python3
"""
calculate accurace
"""


import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    returns a tensor with the accuracy
    """
    y_pred = tf.math.argmax(y_pred, axis=1)
    y = tf.math.argmax(y, axis=1)
    eq = tf.math.equal(y_pred, y)
    acc = tf.reduce_mean(tf.cast(eq, "float"))
    return acc
