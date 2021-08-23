#!/usr/bin/env python3
"""
calculates the precision for each class
"""


import numpy as np


def precision(confusion):
    """returns matrix with the precisions"""
    return np.asarray([confusion[row][row] / confusion[:, row].sum()
                       for row in range(confusion.shape[0])])
