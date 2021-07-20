#!/usr/bin/env python3
"""
concatenate
"""


def cat_arrays(arr1, arr2):
    """concatenate"""
    arr = arr1[:]
    arr.extend(arr2)
    return arr
