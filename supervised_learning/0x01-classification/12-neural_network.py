#!/usr/bin/env python3
"""
Neural Network class
"""


import numpy as np


class NeuralNetwork:
    """neural network class"""

    def __init__(self, nx, nodes):
        """initialization"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for W1"""
        return self.__W1

    @property
    def W2(self):
        """getter for W2"""
        return self.__W2

    @property
    def b1(self):
        """getter for b1"""
        return self.__b1

    @property
    def b2(self):
        """getter for b2"""
        return self.__b2

    @property
    def A1(self):
        """getter for A1"""
        return self.__A1

    @property
    def A2(self):
        """getter for A2"""
        return self.__A2

    def forward_prop(self, X):
        """calculates forward propagation"""
        fp1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = (1 / (1 + np.exp(-fp1)))
        fp2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = (1 / (1 + np.exp(-fp2)))
        return (self.A1, self.A2)

    def cost(self, Y, A):
        """returns the cost"""
        m = Y.shape[1]
        c = - np.sum((Y * np.log(A)) + (1 - Y) * np.log(1.0000001 - A)) / m
        return c

    def evaluate(self, X, Y):
        """evaluates NN predictions"""
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        pred = np.where(A2 >= 0.5, 1, 0)
        return (pred, cost)
