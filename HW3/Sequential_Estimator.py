import Generator
import numpy as np
import math
mean = float(input("mean: "))
variance = float(input("variance: "))
print(f"Data point source function: N({mean}, {variance})")

m = 0  # output mean
m1 = 0
v = 0  # output variance
v1 = 0
n = 0  # how many nodes
Sum = 0
SumSq = 0
while (True):
    new_datapoint = Generator.Univariate_gaussian(mean, math.sqrt(variance))
    print(f"Add data point: {new_datapoint}")
    n += 1
    m = (m * (n-1) + new_datapoint) / n
    Sum = Sum + new_datapoint
    SumSq = SumSq + new_datapoint ** 2
    v = (SumSq - (Sum * Sum) / n)
    pv = v
    if n > 1:
        pv = v/(n-1)
    print(f"Mean = {m}        Variance = {pv}")
    if n > 2 and abs(m - m1) < 1e-4 and abs((v / (n-1)) - (v1 / (n - 2))) < 1e-4:
        break
    m1 = m
    v1 = v
