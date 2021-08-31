#!/usr/bin/env python3
"""save load"""


import tensorflow.keras as K


def save_weights(network, filename, save_format="h5"):
    """returns None"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """returns None"""
    network.load_weights(filename)
    return None
