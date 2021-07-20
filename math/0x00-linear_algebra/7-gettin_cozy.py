#!/usr/bin/env python3
"""
concatenate
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenate 2d"""
    mat = [sub[:] for sub in mat1]
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        for i in mat2:
            mat.append(i)
    elif axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat2)):
            mat[i].extend(mat2[i])
    else:
        return None
    return mat
