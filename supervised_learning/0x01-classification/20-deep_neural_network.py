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

    def forward_prop(self, X):
        """forward prop"""
        self.__cache["A0"] = X
        for i in range(self.L):
            W = self.weights["W{}".format(i + 1)]
            b = self.weights["b{}".format(i + 1)]
            fp = np.matmul(W, self.cache["A{}".format(i)]) + b
            A = (1 / (1 + np.exp(-fp)))
            self.__cache["A{}".format(i + 1)] = A
        return (A, self.cache)

    def cost(self, Y, A):
        """returns the cost"""
        m = Y.shape[1]
        c = - np.sum((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A)) / m
        return c

    def evaluate(self, X, Y):
        """evaluates neuron's predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return (pred, cost)
