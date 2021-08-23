#!/usr/bin/env python3
"""
calculates the precision for each class
"""


import numpy as np


def precision(confusion):
    """returns matrix with the precisions"""
    classes = confusion.shape[0]
    prec = []
    for i in range(classes):
        corr = 0
        tot = 0
        for j in range(classes):
            if i == j:
                corr += confusion[i][j]
            tot += confusion[i][j]
        prec.append(corr / tot)
    return np.asarray(prec)
