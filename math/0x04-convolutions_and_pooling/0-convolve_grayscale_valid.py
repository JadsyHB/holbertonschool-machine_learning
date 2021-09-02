#!/usr/bin/env python3
"""perform a valid convolution"""


import numpy as np


def convolve_grayscale_valid(images, kernel):
    """return the convolved image"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    conv = np.zeros((m, h - kh + 1, w - kw + 1))
    im = np.arange(m)
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            conv[im, i, j] = np.sum(
                images[im, i:kh+i, j:kw+j] * kernel, axis=(1, 2))
    return conv
