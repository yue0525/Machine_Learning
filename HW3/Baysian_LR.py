import Generator
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv


def visualization(axs, title, w, var, a, point_x, point_y):
    axs.set_title(title)
    axs.set_xlim(-2, 2)
    axs.set_ylim(-20, 20)
    Xs = np.linspace(-2, 2, 30)
    function = np.poly1d(np.flipud(w))
    Ys = function(Xs)
    axs.plot(Xs, Ys, color='black')
    if len(point_x) == 0:  # ground_truth
        print(Xs)
        axs.plot(Xs, Ys + a, color='r')
        axs.plot(Xs, Ys - a, color='r')
    else:
        axs.scatter(point_x, point_y, color='blue', s=10)
        Ys_upper = Xs * 0
        Ys_lower = Xs * 0
        for i in range(30):
            X = np.zeros((1, len(w)))
            for j in range(len(w)):
                X[0][j] = Xs[i] ** j
            Ys_upper[i] = Ys[i] + a + np.dot(np.dot(X, var), X.T).item()
            Ys_lower[i] = Ys[i] - a - np.dot(np.dot(X, var), X.T).item()
        axs.plot(Xs, Ys_upper, color='r')
        axs.plot(Xs, Ys_lower, color='r')
    plt.draw()


b = float(input("b: "))
n = int(input("n: "))
a = float(input("a: "))
w = []
for i in range(n):
    buf = float(input(f"w[{i}]: "))
    w.append(buf)
print(f"w = {w}")
point_x = []
point_y = []
num = 0  # how many nodes
var = np.eye(n)/b
mean = np.zeros((n, 1))
error_var = 1
error_mean = 1
predic_mean = 0
predic_var = 0
# print(var)
# print(mean)
number = 0
# while (error_var > 1e-4 or error_mean > 1e-4):
fig, axs = plt.subplots(2, 2)
visualization(axs[0, 0], "Ground truth", w, var, a, point_x, point_y)
result = open("result.txt", 'w')
while (True):
    number += 1
    x, Y = Generator.Polynomial_basis_linear(n, math.sqrt(a), w)
    # print(x)
    # # print(Y)
    # x = -0.64152
    # Y = 0.19039

    point_x.append(x)
    point_y.append(Y)
    result.write(f"Add data point ({x} , {Y}) :\n\n")
    X = np.zeros((1, n))
    for i in range(n):
        X[0][i] = x ** i
    # break
    # C = aXTX + inverse(var)
    C = a * np.dot(X.T, X) + inv(var)
    # m = inv(C)*(aXTY+inv(var)*mean)
    m = np.dot(inv(C), a*(X.T)*Y+np.dot(inv(var), mean))
    new_mean = np.dot(X, m).item()
    new_var = 1/a + np.dot(X, np.dot((inv(C)), (X.T))).item()
    error_var = abs(new_var - predic_var)
    error_mean = abs(new_mean - predic_mean)
    predic_mean = new_mean
    predic_var = new_var
    mean = m
    var = inv(C)
    result.write('Postirior mean:\n')
    for i in range(n):
        result.write(f"{mean[i][0]}\n")
    result.write('\nPosterior variance:\n')
    for i in range(n):
        for j in range(n):
            result.write(f"{var[i][j]}  ")
        result.write("\n")

    result.write(
        f"\nPredictive distribution ~ N({predic_mean}, {predic_var})\n")
    result.write("\n#####################################################\n")
    # print(error_var)
    w_new = []
    for i in range(n):
        w_new.append(mean[i][0])
    if number == 10:
        visualization(axs[1, 0], "After 10 incomes",
                      w_new, var, a, point_x, point_y)
    elif number == 50:
        visualization(axs[1, 1], "After 50 incomes",
                      w_new, var, a, point_x, point_y)
    if error_var < 1e-4 and number > 50:
        break
visualization(axs[0][1], "Predict result", w_new, var, a, point_x, point_y)
plt.show()
