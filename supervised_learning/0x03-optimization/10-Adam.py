#!/usr/bin/env python3
"""Adam optim algo"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """return adam optim op"""
    op = tf.train.AdamOptimizer(
        alpha, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)
    return op
