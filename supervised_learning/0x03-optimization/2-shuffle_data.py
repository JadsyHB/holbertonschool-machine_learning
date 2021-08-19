#!/usr/bin/env python3
"""shuffles matrices"""


import numpy as np


def shuffle_data(X, Y):
    """returns X and Y shuffled"""
    m = X.shape[0]
    sh = np.random.permutation(m)
    Xsh = X[sh]
    Ysh = Y[sh]
    return (Xsh, Ysh)
