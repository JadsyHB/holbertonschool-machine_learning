#!/usr/bin/env python3
"""l2 reg create layer"""


import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """returns output of new layer"""
    reg = tf.contrib.layers.l2_regularizer(lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    t = tf.layers.Dense(units=n, activation=activation,
                        kernel_initializer=init,
                        kernel_regularizer=reg,
                        )
    return t(prev)
