#!/usr/bin/env python3
"""inception block"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Returns The output of the transition layer and the number
             of filters within the output, respectively
    """
    init = K.initializers.he_normal()
    nfilter = int(nb_filters * compression)

    b1 = K.layers.BatchNormalization()(X)

    r1 = K.layers.Activation('relu')(b1)

    c = K.layers.Conv2D(filters=nfilter,
                        kernel_size=1, padding='same',
                        kernel_initializer=init)(r1)
    avg_pool = K.layers.AveragePooling2D(pool_size=2, strides=2,
                                         padding='same')(c)
    return avg_pool, nfilter
