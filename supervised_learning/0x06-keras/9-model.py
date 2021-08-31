#!/usr/bin/env python3
"""save load"""


import tensorflow.keras as K


def save_model(network, filename):
    """returns None"""
    network.save(filename)
    return None


def load_model(filename):
    """returns None"""
    return K.models.load_model(filename)
