#!/usr/bin/env python3
"""build a NN"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """returns keras model"""
    model = K.Sequential()
    L2 = K.regularizers.l2(lambtha)
    model.add(K.layers.Dense(layers[0], input_shape=(nx,),
                             activation=activations[0],
                             kernel_regularizer=L2,
                             name='dense'))
    for i in range(1, len(layers)):
        model.add(K.layers.Dropout(1 - keep_prob))
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=L2,
                                 name='dense_' + str(i)))
    return model


if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], [
                          'tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)
