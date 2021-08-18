#!/usr/bin/env python3
"""calculates normalization constants of matrix"""


import numpy as np


def normalization_constants(X):
    """returns mean and std dev"""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (mean, std)
