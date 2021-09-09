#!/usr/bin/env python3
"""convolutional forward prop"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1, 1)):
    """Returns: the output of the convolutional layer"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        ph = int(np.ceil((((h_prev - 1) * sh + kh - h_prev) / 2)))
        pw = int(np.ceil((((w_prev - 1) * sw + kw - w_prev) / 2)))
    pw, ph = 0, 0
    img_pad = np.pad(A_prev,
                     pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')
    c_h = int(((h_prev + 2 * ph - kh) / sh) + 1)
    c_w = int(((w_prev + 2 * pw - kw) / sw) + 1)
    conv = np.zeros((m, c_h, c_w, c_new))
    for i in range(c_h):
        for j in range(c_w):
            for k in range(c_new):
                v_s = i * sh
                v_e = v_s + kh
                h_s = j * sw
                h_e = h_s + kw
                sliced = img_pad[:, v_s:v_e, h_s:h_e]
                kernel = W[:, :, :, k]
                conv[:, i, j, k] = np.sum(
                    np.multiply(sliced, kernel), axis=(1, 2, 3))
    return activation(conv + b)
