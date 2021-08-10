#!/usr/bin/env python3
"""
deep neural network
"""


import numpy as np


class DeepNeuralNetwork:
    """
    deep neural network class
    """

    def __init__(self, nx, layers):
        """
        initialization
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        weights = {}
        pv = nx
        for i in range(len(layers)):
            if type(layers[i]) is not int or layers[i] < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            weights["W{}".format(i + 1)] = (
                np.random.randn(layers[i], pv) * np.sqrt(2 / pv)
            )
            pv = layers[i]
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """getter for L"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for weights"""
        return self.__weights
