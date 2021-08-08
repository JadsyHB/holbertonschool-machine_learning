#!/usr/bin/env python3
"""
Neuron class
"""


import matplotlib.pyplot as plt
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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]
        dz = A - Y
        dW = (1 / m) * np.matmul(X, dz.transpose()).transpose()
        db = (1 / m) * np.sum(dz)
        self.__W = self.W - (alpha * dW)
        self.__b = self.b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        pts = []
        for i in range(iterations):
            A = self.forward_prop(X)
            if verbose and i % step == 0:
                c = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, c))
            if graph and i % step == 0:
                c = self.cost(Y, A)
                pts.append(c)
            self.gradient_descent(X, Y, A, alpha)
        if verbose:
            c = self.cost(Y, A)
            print("Cost after {} iterations: {}".format(iterations, c))
        if graph:
            c = self.cost(Y, A)
            pts.append(c)
            x = np.arange(0, iterations + 1, step)
            y = np.asarray(pts)
            plt.plot(x, y, 'blue')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)
