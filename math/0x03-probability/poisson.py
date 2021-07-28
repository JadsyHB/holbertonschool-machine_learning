#!/usr/bin/env python3
"""
poissom distribution class probabiolity module
"""


class Poisson:
    """Poisson distribution probability class"""
    def __init__(self, data=None, lambtha=1.):
        """initialization"""
        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """pmf function"""
        if k < 0:
            return 0
        k = int(k)
            fact = 1
        for i in range(1, k+1):
            fact *= i
        p = ((2.7182818285**-(self.lambtha))*(self.lambtha**k)) / fact
        return p

    def cdf(self, k):
        """cdf function"""
        if k < 0:
            return 0
        k = int(k)
        p = 0
        for i in range(k+1):
            p += self.pmf(i)
        return p
