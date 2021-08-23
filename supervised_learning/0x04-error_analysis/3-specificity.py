#!/usr/bin/env python3
"""
calculates specificity
"""


import numpy as np


def specificity(confusion):
    """returns matrix with specificities"""
    classes = confusion.shape[0]
    spec = [np.zeros(classes)]
    total = sum(confusion)
    for i in range(classes):
        val = confusion[i, i]
        fp = sum(confusion[:, i]) - val
        tn = sum(total) - sum(confusion[:, i]) - sum(confusion[i, :]) + val
        spec[i] = tn / (tn + fp)
    return spec
