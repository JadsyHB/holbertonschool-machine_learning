#!/usr/bin/env python3
"""convert to one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """returns one hot matrix"""
    oh = np.zeros((Y.shape[0], classes))
    oh[np.arange(Y.shape[0]), Y] = 1
    return oh
