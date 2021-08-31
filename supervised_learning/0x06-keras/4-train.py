#!/usr/bin/env python3
"""train using mini batch gradient desc"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """returns the history"""
    hist = network.fit(data, labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle)
    return hist
