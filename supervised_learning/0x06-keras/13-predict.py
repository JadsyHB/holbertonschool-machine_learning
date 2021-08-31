#!/usr/bin/env python3
"""predict a DNN"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Returns the prediction of the data
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction
