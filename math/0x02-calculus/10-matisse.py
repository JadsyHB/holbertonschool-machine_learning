#!/usr/bin/env python3
"""poly deriv"""

def poly_derivative(poly):
    """poly deriv"""
    if poly:
        deriv = []
        for i in range(len(poly)):
            deriv.append(i*poly[i])
        return deriv[1:]
    return None
