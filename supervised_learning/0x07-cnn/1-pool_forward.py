#!/usr/bin/env python3
"""convolutional forward prop"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Returns: the output of the pooling layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    c_h = int((h_prev - kh) / sh) + 1
    c_w = int((w_prev - kw) / sw) + 1

    conv = np.zeros((m, c_h, c_w, c_prev))
    for i in range(c_h):
        for j in range(c_w):
            if mode == "max":
                conv[:, i, j] = np.max(
                    A_prev[:, i * sh:((i * sh) + kh),
                           j * sw:((j * sw) + kw)], axis=(1, 2))
            elif mode == "avg":
                conv[:, i, j] = np.mean(
                    A_prev[:, i * sh:((i * sh) + kh),
                           j * sw:((j * sw) + kw)], axis=(1, 2))
    return conv
