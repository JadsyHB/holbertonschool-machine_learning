#!/usr/bin/env python3
"""summation square"""


def summation_i_squared(n):
    """
    returns sum squared
    """
    try:
        return int((n*(n+1)*(2*n+1))/6)
    except:
        return None
