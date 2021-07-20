#!/usr/bin/env python3
"""summation square"""

def summation_i_squared(n):
    """
    returns sum squared
    """
    if n:
        return int((n*(n+1)*(2*n+1))/6)
    return None
