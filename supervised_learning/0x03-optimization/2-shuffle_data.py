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


X = np.array([[1, 2],
              [3, 4],
              [5, 6],
              [7, 8],
              [9, 10]])
Y = np.array([[11, 12],
              [13, 14],
              [15, 16],
              [17, 18],
              [19, 20]])
