#!/usr/bin/env python3
"""identity network"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """Returns: the activated output of the identity block"""
    init = K.initializers.he_normal()
    activation = 'relu'
    F11, F3, F12 = filters

    c_1 = K.layers.Conv2D(filters=F11, kernel_size=1, padding='same',
                          kernel_initializer=init)(A_prev)

    b1 = K.layers.BatchNormalization(axis=3)(c_1)

    r1 = K.layers.Activation('relu')(b1)

    c_2 = K.layers.Conv2D(filters=F3, kernel_size=3, padding='same',
                          kernel_initializer=init)(r1)

    b2 = K.layers.BatchNormalization(axis=3)(c_2)

    r2 = K.layers.Activation('relu')(b2)

    c_3 = K.layers.Conv2D(filters=F12, kernel_size=1, padding='same',
                          kernel_initializer=init)(r2)

    b3 = K.layers.BatchNormalization(axis=3)(c_3)

    add = K.layers.Add()([b3, A_prev])

    fr = K.layers.Activation('relu')(add)

    return fr
