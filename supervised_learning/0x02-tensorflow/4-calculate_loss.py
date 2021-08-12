#!/usr/bin/env python3
"""
calculates loss
"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    returns tensor containing loss
    """
    loss = tf.losses.softmax_cross_entropy(y, logits=y_pred)
    return loss
