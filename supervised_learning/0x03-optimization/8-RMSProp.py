#!/usr/bin/env python3
"""RMSProp optim algo"""


import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """returns RMSProp optim"""
    op = tf.train.RMSPropOptimizer(
        alpha, decay=beta2, epsilon=epsilon).minimize(loss)
    return op
