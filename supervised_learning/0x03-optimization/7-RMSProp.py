#!/usr/bin/env python3
"""RMSProp optim algo"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """returns updated variable and the new moment"""
    sdw = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var -= alpha * (grad / (epsilon + (sdw ** (1/2))))
    return var, sdw
