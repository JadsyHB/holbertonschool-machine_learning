#!/usr/bin/env python3
"""perform a valid convolution"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """return convolved images"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    pad_w, pad_h = padding[1], padding[0]
    out_h = h + (2 * pad_h) - kh + 1
    out_w = w + (2 * pad_w) - kw + 1
    im_pad = np.pad(images, pad_width=((0, 0), (pad_h, pad_h),
                                       (pad_w, pad_w)), mode='constant')
    conv = np.zeros((m, out_h, out_w))
    im = np.arrange(m)
    for i in range(out_h):
        for j in range(out_w):
            conv[im, i, j] = np.sum(
                im_pad[im, i:kh+i, j:kw+j] * kernel, axis=(1, 2))
    return conv
