#!/usr/bin/env python3
"""test a DNN"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Returns the loss and accuracy
    """
    eval = network.evaluate(data, labels, verbose=verbose)
    return eval
