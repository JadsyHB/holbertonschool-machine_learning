#!/usr/bin/env python3
"""l2 reg cost"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """returns cost of network L2 reg"""
    w = 0
    for i in range(1, L + 1):
        lw = weights["W{}".format(i)]
        w += np.linalg.norm(lw)
    l2 = cost + ((lambtha / (2 * m)) * w)
    return l2
