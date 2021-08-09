#!/usr/bin/env python3
"""
Neural Network class
"""


import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """calculates gradient descent"""
        m = Y.shape[1]
        dz2 = A2 - Y
        dW2 = (1 / m) * np.matmul(dz2, A1.transpose())
        db2 = (1 / m) * (np.sum(dz2, axis=1, keepdims=True))
        dz1 = (np.matmul(self.W2.transpose(), dz2)) * (A1 * (1 - A1))
        dW1 = (1 / m) * (np.matmul(dz1, X.transpose()))
        db1 = (1 / m) * (np.sum(dz1, axis=1, keepdims=True))
        self.__W1 = self.W1 - (alpha * dW1)
        self.__b1 = self.b1 - (alpha * db1)
        self.__W2 = self.W2 - (alpha * dW2)
        self.__b2 = self.b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """trains neural network"""
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
            A1, A2 = self.forward_prop(X)
            if verbose and i % step == 0:
                c = self.cost(Y, A2)
                print("Cost after {} iterations: {}".format(i, c))
            if graph and i % step == 0:
                c = self.cost(Y, A2)
                pts.append(c)
            self.gradient_descent(X, Y, A1, A2, alpha)
        if verbose:
            c = self.cost(Y, A2)
            print("Cost after {} iterations: {}".format(iterations, c))
        if graph:
            c = self.cost(Y, A2)
            pts.append(c)
            x = np.arange(0, iterations + 1, step)
            y = np.asarray(pts)
            plt.plot(x, y, 'blue')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return self.evaluate(X, Y)


lib_train = np.load('Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
nn = NeuralNetwork(X_train.shape[0], 3)
A, cost = nn.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
