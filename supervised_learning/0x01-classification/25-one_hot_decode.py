#!/usr/bin/env python3
"""convert from one-hot matrix"""


import numpy as np


def one_hot_decode(one_hot):
    """converts from one hot matrix"""
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    return one_hot.transpose().argmax(axis=1)
