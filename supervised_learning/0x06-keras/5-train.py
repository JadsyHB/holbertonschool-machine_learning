#!/usr/bin/env python3
"""train using mini batch gradient desc"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """validates data or None"""
    val = network.fit(x=data, y=labels, batch_size=batch_size,
                      epochs=epochs,
                      validation_data=validation_data,
                      verbose=verbose,
                      shuffle=shuffle)
    return val
