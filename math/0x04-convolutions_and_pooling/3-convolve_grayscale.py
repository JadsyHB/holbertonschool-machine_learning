#!/usr/bin/env python3
"""perform a valid convolution"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """return convolved images"""
    m, h, w = images.shape[0], images.shape[1], images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]
    pad_w, pad_h = 0, 0
    sh, sw = stride[0], stride[1]
    if padding == "same":
        pad_h = int((((h - 1) * sh + kh - h) / 2) + 1)
        pad_w = int((((w - 1) * sw + kw - w) / 2) + 1)
    if type(padding) is tuple:
        pad_h, pad_w = padding[0], padding[1]
    out_h = int(((h + 2 * pad_h - kh) / sh) + 1)
    out_w = int(((w + 2 * pad_w - kh) / sw) + 1)
    im_pad = np.pad(images, pad_width=((0, 0), (pad_h, pad_h),
                                       (pad_w, pad_w)), mode='constant')
    conv = np.zeros((m, out_h, out_w))
    im = np.arange(m)
    for i in range(out_h):
        for j in range(out_w):
            conv[im, i, j] = np.sum(
                im_pad[im, i * sh:((i * sh) + kh),
                       j * sw:((j * sw) + kw)] * kernel, axis=(1, 2))
    return conv
