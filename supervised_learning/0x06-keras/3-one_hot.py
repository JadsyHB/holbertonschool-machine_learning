#!/usr/bin/env python3
"""convert vector to oh"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """returns oh matrix"""
    oh = K.utils.to_categorical(labels, num_classes=classes)
    return oh
