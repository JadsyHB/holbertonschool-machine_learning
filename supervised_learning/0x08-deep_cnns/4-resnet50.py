#!/usr/bin/env python3
"""resnet50"""

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Returns: the keras model"""
    X = K.Input(shape=(224, 224, 3))
    init = K.initializers.he_normal()

    c_1 = K.layers.Conv2D(filters=64, kernel_size=7, padding='same',
                          strides=2, kernel_initializer=init)(X)

    b1 = K.layers.BatchNormalization()(c_1)

    r1 = K.layers.Activation('relu')(b1)

    p1 = K.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(r1)

    prjc_1 = projection_block(p1, [64, 64, 256], 1)

    idc_2_2 = identity_block(prjc_1, [64, 64, 256])
    idc_2_3 = identity_block(idc_2_2, [64, 64, 256])

    prjc_2 = projection_block(idc_2_3, [128, 128, 512])

    idconv3_1 = identity_block(prjc_2, [128, 128, 512])
    idconv3_2 = identity_block(idconv3_1, [128, 128, 512])
    idconv3_3 = identity_block(idconv3_2, [128, 128, 512])

    prjconv3 = projection_block(idconv3_3, [256, 256, 1024])

    idconv4_1 = identity_block(prjconv3, [256, 256, 1024])
    idconv4_2 = identity_block(idconv4_1, [256, 256, 1024])
    idconv4_3 = identity_block(idconv4_2, [256, 256, 1024])
    idconv4_4 = identity_block(idconv4_3, [256, 256, 1024])
    idconv4_5 = identity_block(idconv4_4, [256, 256, 1024])

    prjconv4 = projection_block(idconv4_5, [512, 512, 2048])

    idconv5_1 = identity_block(prjconv4, [512, 512, 2048])
    idconv5_2 = identity_block(idconv5_1, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D(pool_size=7,
                                         padding='same')(idconv5_2)

    FC = K.layers.Dense(1000, activation='softmax',
                        kernel_initializer=init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=FC)

    return model