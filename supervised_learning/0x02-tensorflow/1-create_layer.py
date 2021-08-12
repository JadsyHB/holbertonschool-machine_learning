#!/usr/bin/env python3
"""
create placeholders module
"""


import tensorflow as tf


def create_layer(prev, n, activation):
    """returns tensor output"""
    init_w = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernel_initializer=init_w
    )
    return layer(prev)
