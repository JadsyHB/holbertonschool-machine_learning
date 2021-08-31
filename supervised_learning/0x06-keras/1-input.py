#!/usr/bin/env python3
"""build a NN"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """returns keras model"""
    inputs = K.Input(shape=(nx,))
    L2 = K.regularizers.l2(lambtha)
    out = K.layers.Dense(layers[0],
                         activation=activations[0],
                         kernel_regularizer=L2)(inputs)
    for i in range(1, len(layers)):
        dp = K.layers.Dropout(1 - keep_prob)(out)
        out = K.layers.Dense(layers[i], activation=activations[i],
                             kernel_regularizer=L2)(dp)
    return K.models.Model(inputs=inputs, outputs=out)
