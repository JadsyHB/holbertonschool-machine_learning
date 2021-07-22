#!/usr/bin/env python3
"""
integration of a polynomial
"""


def poly_integral(poly, C=0):
    """
    returns list of coefficients
    """
    if type(poly) is not list or len(poly) < 1:
        return None
    if type(C) is not int and type(C) is not float:
        return None
    for i in poly:
        if type(i) is not int and type(i) is not float:
            return None
    if type(C) is float:
        C = int(C)
    integral = [C]
    for i in range(len(poly)):
        if (poly[i] % (i + 1)) == 0:
            coef = poly[i] // (i + 1)
        else:
            coef = poly[i] / (i + 1)
        integral.append(coef)
    while integral[-1] is 0 and len(integral) > 1:
        integral = integral[:-1]
    return integral
