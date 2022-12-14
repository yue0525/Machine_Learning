from Generator import Univariate_gaussian

import numpy as np
import math
from matplotlib import pyplot


def drawPlot(title, N, points, labels):

    pyplot.title(title)
    for i in range(N):
        if labels[i] == 0:
            pyplot.scatter(points[i][0], points[i][1], color='red', s=10)
        else:
            pyplot.scatter(points[i][0], points[i][1], color='blue', s=10)
    pyplot.draw()


def printResult(W, labels, predict, judge):
    result = ""
    if judge == 0:
        result += f"Gradient descent:\n\n"
    if judge == 1:
        result += f"\nNewton's method:\n"
    result += f"w:\n"
    for w in W:
        result += f"{w}\n"
    buf = np.zeros((2, 2), dtype=int)
    for i in range(len(labels)):
        buf[labels[i]][predict[i]] += 1
    result += f"\nConfusion Matrix:\n"
    result += f"\t\t\t\t Predict cluster 1\t Predict cluster 2\n"
    result += f"Is cluster 1\t     {buf[0][0]} \t\t\t    {buf[0][1]}\n"
    result += f"Is cluster 2\t     {buf[1][0]} \t\t\t    {buf[1][1]}\n"
    Sensitivity = buf[0][0]/sum(buf[0])
    result += f"Sensitivity (Successfully predict cluster 1): {Sensitivity}\n"
    Specificity = buf[1][1]/sum(buf[1])
    result += f"Specificity (Successfully predict cluster 2): {Specificity}\n"
    return result


def Gradient_descent(X, Y):
    W = np.random.rand(3, 1)
    count = 0
    while (True):
        count += 1
        W_pre = W
        Buf = X.T.dot(Y - 1 / (1 + np.exp(-X.dot(W))))
        W = Buf + W_pre
        if (count > 100000 or np.linalg.norm(W - W_pre) < 1e-2):
            break
    return W


def Newton(X, Y):
    W = np.random.rand(3, 1)
    D = np.zeros((N*2, N*2))
    count = 0
    while (True):
        count += 1
        gradient = X.T.dot(Y - 1 / (1 + np.exp(-X.dot(W))))
        for i in range(N*2):
            buffer_up = np.exp(-X[i].dot(W))
            if math.isinf(buffer_up):
                buffer_up = np.exp(100)
            buffer_down = (buffer_up + 1)**2
            if buffer_down == 0:
                buffer_down = buffer_up
            D[i][i] = buffer_up/buffer_down
        H = X.T.dot(D.dot(X))
        W_pre = W
        gradient = np.array(gradient)
        if np.linalg.det(H) == 0:
            W = W + gradient
        else:
            W = W + np.linalg.inv(H).dot(gradient)
        if (count > 100000 or np.linalg.norm(W - W_pre) < 1e-2):
            break

    return W


def prediction(W, X):
    predictionS = []
    for i in range(len(X)):
        if X[i].dot(W) >= 0:
            predictionS.append(1)
        else:
            predictionS.append(0)

    return np.array([predictionS], dtype=int).T


if __name__ == "__main__":
    N = int(input("N: "))
    mx1 = float(input("mx1: "))
    my1 = float(input("my1: "))
    mx2 = float(input("mx2: "))
    my2 = float(input("my2: "))
    vx1 = float(input("vx1: "))
    vy1 = float(input("vy1: "))
    vx2 = float(input("vx1: "))
    vy2 = float(input("vy1: "))
    D1 = []
    D2 = []
    points = []
    for label in range(N):
        point = [Univariate_gaussian(mx1, vx1), Univariate_gaussian(my1, vy1)]
        D1.append(point)
        point = [Univariate_gaussian(mx2, vx2), Univariate_gaussian(my2, vy2)]
        D2.append(point)
    points = D1 + D2
    X = np.ones((2*N, 3))
    X[0:N, 0:2] = D1
    X[N:2*N, 0:2] = D2
    Y = np.zeros((2*N, 1), dtype=int)
    Y[N:2*N, 0] = 1
    resultFile = open(f"result_1.txt", 'w')
    pyplot.figure()
    pyplot.subplot(131)
    drawPlot("Ground truth", 2*N, points, Y.flatten())

    W = Gradient_descent(X, Y)
    predict = prediction(W, X)
    result = printResult(W.flatten(), Y.flatten(), predict.flatten(), 0)
    pyplot.subplot(132)
    drawPlot("Gradient descent", 2*N, points, predict.flatten())

    W = Newton(X, Y)
    predict = prediction(W, X)
    result += printResult(W.flatten(), Y.flatten(), predict.flatten(), 1)
    pyplot.subplot(133)
    drawPlot("Newton's method", 2*N, points, predict.flatten())

    pyplot.show()
    resultFile.write(result)
