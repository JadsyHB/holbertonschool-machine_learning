#!/usr/bin/env python3

"""Transpose a matrix"""


def matrix_transpose(matrix):
    """Transpose a matrix"""
    m = matrix
    result = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    return result
