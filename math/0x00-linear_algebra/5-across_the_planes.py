#!/usr/bin/env python3
"""
2d add
"""


def add_matrices2D(mat1, mat2):
    """adding matrices"""
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    sum = []
    for i in range(len(mat1)):
        inner_sum = []
        for j in range(len(mat1[0])):
            inner_sum.append(mat1[i][j] + mat2[i][j])
        sum.append(inner_sum)
    return sum
