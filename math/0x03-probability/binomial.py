#!/usr/bin/env python3
"""
binomial distribution probability class py
"""


class Binomial:
    """Binomial distribution class"""
    def __init__(self, data=None, n=1, p=0.5):
        """initialization"""
        self.n = int(n)
        self.p = float(p)
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / int(len(data))
            l = []
            for i in data:
                l.append((i - mean) ** 2)
            v = sum(l) / len(l)
            t = 1 - (v / mean)
            self.n = int(round(mean / t))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """returns pmf"""
        if k < 0 or k > self.n:
            return 0
        k = int(k)
        f1 = 1
        f2 = 1
        f3 = 1
        for i in range(1, self.n + 1):
            f1 *= i
        for j in range(1, k + 1):
            f2 *= j
        for l in range(1, self.n - k + 1):
            f3 *= l
        comb = f1 / (f2 * f3)
        p = comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))
        return p

    def cdf(self, k):
        """returns cdf"""
        if k < 0:
            return 0
        p = 0
        k = int(k)
        for i in range(k + 1):
            p += self.pmf(i)
        return p
