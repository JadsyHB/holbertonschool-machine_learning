#!/usr/bin/env python3
"""inception network"""

import tensorflow.keras as K

inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """return the keras model"""
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()
    active = 'relu'

    c_1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=(
        2, 2), padding='same', activation=active, kernel_initializer=init)(X)
    mp1 = K.layers.MaxPooling2D(
        pool_size=[3, 3], strides=(2, 2), padding='same')(c_1)
    c_2 = K.layers.Conv2D(filters=64, kernel_size=1, padding='valid',
                          activation=active, kernel_initializer=init)(mp1)
    cd_2 = K.layers.Conv2D(filters=192, kernel_size=3, padding='same',
                           activation=active,
                           kernel_initializer=init)(c_2)
    mp2 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                padding='same')(cd_2)
    ib_1 = inception_block(mp2, [64, 96, 128, 16, 32, 32])
    ib_2 = inception_block(ib_1, [128, 128, 192, 32, 96, 64])

    max_pool_3 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(ib_2)

    ib_3 = inception_block(max_pool_3, [192, 96, 208, 16, 48, 64])
    ib_4 = inception_block(ib_3, [160, 112, 224, 24, 64, 64])
    ib_5 = inception_block(ib_4, [128, 128, 256, 24, 64, 64])
    ib_6 = inception_block(ib_5, [112, 144, 288, 32, 64, 64])
    ib_7 = inception_block(ib_6, [256, 160, 320, 32, 128, 128])

    max_pool_4 = K.layers.MaxPooling2D(pool_size=[3, 3], strides=(2, 2),
                                       padding='same')(ib_7)

    ib_8 = inception_block(max_pool_4, [256, 160, 320, 32, 128, 128])
    ib_9 = inception_block(ib_8, [384, 192, 384, 48, 128, 128])

    avg_pool = K.layers.AveragePooling2D(pool_size=[7, 7], strides=(1, 1),
                                         padding='valid')(ib_9)

    dropout = K.layers.Dropout(.4)(avg_pool)

    FC = K.layers.Dense(1000, activation='softmax',
                        kernel_initializer=init)(dropout)

    model = K.models.Model(inputs=X, outputs=FC)

    return model
