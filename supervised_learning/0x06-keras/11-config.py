#!/usr/bin/env python3
"""save load"""


import tensorflow.keras as K


def save_config(network, filename):
    """returns None"""
    with open(filename, 'w') as f:
        f.write(network.to_json())
    return None


def load_config(filename):
    """returns None"""
    with open(filename, 'r') as f:
        network_config = f.read()
    return K.models.model_from_json(network_config)
