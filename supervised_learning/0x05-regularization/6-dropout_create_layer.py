#!/usr/bin/env python3
"""dropout in tf"""


import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """returns output of new layer"""
    do = tf.layers.Dropout(keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    t = tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=init,
                        kernel_regularizer=do)
    return t(prev)
