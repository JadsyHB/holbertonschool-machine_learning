#!/usr/bin/env python3
"""
calculates specificity
"""


import numpy as np


def specificity(confusion):
    """returns matrix with specificities"""
    classes = confusion.shape[0]
    spec = [np.delete(np.delete(confusion, i, 1), i, 0).sum(
    ) / np.delete(confusion, i, 0).sum() for i in range(classes)]
    return np.asarray(spec)
