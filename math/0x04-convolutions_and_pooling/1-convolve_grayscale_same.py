#!/usr/bin/env python3
"""perform a valid convolution"""


import numpy as np


def convolve_grayscale_same(images, kernel):
    """return convolved images"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    h_ = int((kh - 1) / 2)
    w_ = int((kw - 1) / 2)
    if kh % 2 == 0:
        h_ = int(kh / 2)
    if kw % 2 == 0:
        w_ = int(kw / 2)
    im_pad = np.pad(images, pad_width=(
        (0, 0), (h_, h_), (w_, w_)), mode="constant")
    conv = np.zeros((m, h, w))
    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            conv[im_pad, i, j] = np.sum(
                images[im_pad, i:kh+i, j:kw+j] * kernel, axis=(1, 2))
    return conv
