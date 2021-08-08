#!/usr/bin/env python3
"""
Neuron class
"""


import numpy as np


class Neuron:
    """
    Neuron class
    """

    def __init__(self, nx):
        """
        initialization
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for W"""
        return self.__W

    @property
    def b(self):
        """getter for b"""
        return self.__b

    @property
    def A(self):
        """getter for A"""
        return self.__A

    def forward_prop(self, X):
        """calculates the forward propagation"""
        fp = np.matmul(self.W, X) + self.b
        sig = (1 / (1 + np.exp(-fp)))
        self.__A = sig
        return self.A

    def cost(self, Y, A):
        """returns the cost"""
        m = Y.shape[1]
        c = - np.sum((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A)) / m
        return c

    def evaluate(self, X, Y):
        """evaluates neuron's predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return (pred, cost)
