#!/usr/bin/env python3
"""calculates moving avg"""

import matplotlib.pyplot as plt
import numpy as np


def moving_average(data, beta):
    """return list containing moving avgs"""
    a = 0
    mavg = []
    for i in range(len(data)):
        a = (a * beta) + ((1 - beta) * data[i])
        mavg.append(a / (1 - (beta ** (i+1))))
    return mavg
