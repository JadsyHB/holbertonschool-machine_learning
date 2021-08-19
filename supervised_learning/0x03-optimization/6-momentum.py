#!/usr/bin/env python3
"""trains using gradient descent with mom"""


import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """returns the momentum optim operation"""
    op = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
    return op
