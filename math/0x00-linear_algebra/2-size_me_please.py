#!/usr/bin/env python3
"""
dimensions of matrrix
"""


def matrix_shape(matrix):
    """returns list of dimensions"""
    dim = []
    if type(matrix) is list:
        dim.append(len(matrix))
        dim.extend(matrix_shape(matrix[0]))

    return dim
