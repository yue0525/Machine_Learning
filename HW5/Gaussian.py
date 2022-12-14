import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def load():
    f = open("HW5\input.data", "r", encoding='utf-8')
    X = []
    Y = []
    for line in f:
        x, y = line.split(' ')
        X.append(float(x))
        Y.append(float(y))
    X = np.array(X, dtype=np.float64).reshape(-1, 1)
    Y = np.array(Y, dtype=np.float64).reshape(-1, 1)
    return X, Y


def rational_quadratic_kernal(Xa, Xb, variance, alpha, l):
    distance = np.empty((len(Xa), len(Xb)))
    for i in range(len(Xa)):
        for j in range(len(Xb)):
            distance[i][j] = (Xa[i] - Xb[j])**2
    kernal = variance * ((1+distance/2*alpha*l**2)**(-1*alpha))
    # print(kernal)
    return kernal


def guassian(X, Y, beta, variance, alpha, length_scale):
    kernal = rational_quadratic_kernal(X, X, variance, alpha, length_scale)
    # print("kernal")
    C = kernal + np.eye(len(X))/beta
    C_inv = np.linalg.inv(C)
    X_star = np.linspace(-60, 60, 1000).reshape(-1, 1)
    # print("----------------")
    # print(X_star)
    kernal_x_xstar = rational_quadratic_kernal(
        X, X_star, variance, alpha, length_scale)
    mean_x_star = kernal_x_xstar.T.dot(C_inv).dot(Y)
    kernal_star = rational_quadratic_kernal(
        X_star, X_star, variance, alpha, length_scale) + np.eye(len(X_star))/beta
    variance_x_star = kernal_star - \
        kernal_x_xstar.T.dot(C_inv).dot(kernal_x_xstar)
    return mean_x_star, variance_x_star, X_star


def visualization(X, Y, mean, var, x_star, title, beta, alpha, sigma, length_scale):
    plt.scatter(X, Y, color='black', s=10, zorder=15)
    interval = 2 * np.sqrt(var)
    # print("inter : ", interval)
    plt.plot(x_star, mean + interval, color='pink', zorder=5)
    plt.plot(x_star, mean - interval, color='pink', zorder=5)
    plt.plot(x_star, mean, color='blue', zorder=10)
    # x_star = x_star.reshape(1, -1)
    # print(x_star.shape)
    plt.xlim(-60, 60)
    plt.title(
        f'beta:{beta} sigma: {sigma:.4f}, alpha: {alpha:.4f}, length scale: {length_scale:.4f}')
    plt.show()


def fun(theta, data_X, data_Y, beta):
    theta = theta.ravel()
    # print(theta)
    kernal = rational_quadratic_kernal(
        data_X, data_X, theta[0], theta[1], theta[2])
    C = kernal + np.eye(len(data_X))/beta
    C_inv = np.linalg.inv(C)
    negative = 0.5 * np.log(2 * np.pi)
    negative += 0.5 * np.log(np.linalg.det(C))
    negative += 0.5 * (data_Y.T.dot(C_inv).dot(data_Y))
    return negative


if __name__ == "__main__":

    data_X, data_Y = load()
    beta = 5
    length_scale = 1
    variance = 1
    alpha = 1
    mean, var, x_star = guassian(
        data_X, data_Y, beta, variance, alpha, length_scale)

    visualization(data_X, data_Y, mean, var, x_star, "normal",
                  beta, alpha, variance, length_scale)

    opt = minimize(
        fun, [variance, alpha, length_scale],
        args=(data_X, data_Y, beta),
        bounds=((1e-8, 1e8), (1e-8, 1e8), (1e-8, 1e8)))

    variance_opt = opt.x[0]
    alpha_opt = opt.x[1]
    length_scale_opt = opt.x[2]

    mean, var, x_star = guassian(
        data_X, data_Y, beta, variance_opt, alpha_opt, length_scale_opt)

    visualization(data_X, data_Y, mean, var, x_star, "optimization",
                  beta, alpha_opt, variance_opt, length_scale_opt)
