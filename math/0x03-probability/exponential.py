#!/usr/bin/env python3
"""
exponential distribution class probabiolity module
"""


class Exponential:
    """exponential distribution probability class"""
    def __init__(self, data=None, lambtha=1.):
        """initialization"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, x):
        """pdf function"""
        if x < 0:
            return 0
        fact = 1
        p = self.lambtha*(2.7182818285**-(self.lambtha * x))
        return p

    def cdf(self, x):
        """cdf function"""
        if x < 0:
            return 0
        p = 1 - (2.7182818285**-(self.lambtha * x))
        return p
