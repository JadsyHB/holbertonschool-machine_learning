#!/usr/bin/env python3
"""
calculates the sensitivity for each class
"""


import numpy as np


def sensitivity(confusion):
    """returns a matrix containing the sensitivity"""
    classes = confusion.shape[0]
    sens = []
    for i in range(classes):
        corr = 0
        tot = 0
        for j in range(classes):
            if i == j:
                corr += confusion[i][j]
            tot += confusion[i][j]
        sens.append(corr / tot)
    return np.asarray(sens)
