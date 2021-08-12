#!/usr/bin/env python3
"""
train using gradient descent
"""


import tensorflow as tf


def create_train_op(loss, alpha):
    """
    trains NN
    """
    op = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return op
