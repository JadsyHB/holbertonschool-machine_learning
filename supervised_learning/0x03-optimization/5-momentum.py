#!/usr/bin/env python3
"""update variables with momentum optim of grad"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """returns updated variable and new moment"""
    dw = (beta1 * v) + ((1 - beta1) * grad)
    var -= alpha * dw
    return var, dw
