import numpy as np


def Univariate_gaussian(m, s):
    datapoint = (sum(np.random.uniform(0, 1, 12))-6)*s + m
    return datapoint


def Polynomial_basis_linear(n, a, w):  # n is the polynomial power
    e = Univariate_gaussian(0, a)
    x = np.random.uniform(-1.0, 1.0)
    for i in range(len(w)):
        # print(w[i])
        e += w[i]*(x**i)
    return x, e


# print(np.random.uniform(0, 1, 12))
