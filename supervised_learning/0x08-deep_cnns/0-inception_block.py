#!/usr/bin/env python3
"""inception block"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Returns the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    init = K.initializers.he_normal(seed=None)
    active = 'relu'

    c_1 = K.layers.Conv2D(filters=F1, kernel_size=1, padding='same',
                          activation=active, kernel_initializer=init)(A_prev)
    c_2 = K.layers.Conv2D(filters=F3R, kernel_size=1, padding='same',
                          activation=active, kernel_initializer=init)(A_prev)
    c_3 = K.layers.Conv2D(filters=F5R, kernel_size=1, padding='same',
                          activation=active, kernel_initializer=init)(A_prev)
    dc_1 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                           activation=active, kernel_initializer=init)(c_2)
    dc_2 = K.layers.Conv2D(filters=F5, kernel_size=5, padding='same',
                           activation=active, kernel_initializer=init)(c_3)
    lp = K.layers.MaxPooling2D(
        pool_size=[3, 3], strides=(1, 1), padding='same')(A_prev)
    lpP = K.layers.Conv2D(filters=FPP, kernel_size=1, padding='same',
                          activation=active, kernel_initializer=init)(lp)
    mid = K.layers.concatenate([c_1, dc_1, dc_2, lpP])
    return mid
