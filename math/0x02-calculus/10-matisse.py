#!/usr/bin/env python3
"""poly deriv"""


def poly_derivative(poly):
    """poly deriv"""
    try:
        deriv = []
        for i in range(len(poly)):
            deriv.append(i*poly[i])
        return deriv[1:]
    except:
        return None
