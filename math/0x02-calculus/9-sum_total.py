#!/usr/bin/env python3
"""summation square of a number n without looping"""


def summation_i_squared(n):
    """
    returns sum squared without looping
    """
    if type(n) is not int or n < 1:
        return None
    return int((n * (n + 1) * (2 * n + 1)) / 6)
