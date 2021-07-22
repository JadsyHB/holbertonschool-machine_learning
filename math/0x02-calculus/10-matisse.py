#!/usr/bin/env python3
"""poly deriv returning a list of coeffs"""


def poly_derivative(poly):
    """poly deriv returns a list of coeffs"""
     if type(poly) is not list or len(poly) < 1:
         return None
     for i in poly:
         if type(i) is not int and type(i) is not float:
             return None
    deriv = []
    for i in range(len(poly)):
        deriv.append(i * poly[i])
    while deriv[-1] is 0 and len(deriv) > 1:
        deriv = deriv[:-1]
    return derivative
