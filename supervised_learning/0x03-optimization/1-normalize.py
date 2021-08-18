#!/usr/bin/env python3
"""normalizes matrix"""


import numpy as np


def normalize(X, m, s):
    """returns normalized matrix"""
    norm = (X - m) / s
    return norm
