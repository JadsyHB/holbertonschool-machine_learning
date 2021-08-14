#!/usr/bin/env python3
"""convert to one-hot matrix"""


import numpy as np


def one_hot_encode(Y, classes):
    """returns one hot matrix"""
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    oh = np.eye(classes)[Y].transpose()
    return oh
