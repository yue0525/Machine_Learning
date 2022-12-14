import numpy as np
import csv
from libsvm.svmutil import *
# from matplotlib import pyplot
import time
from scipy.spatial.distance import cdist


def load():
    # x_train_buf=[]
    y_train_buf1 = []
    y_test_buf1 = []
    x_train_list = []
    x_test_list = []

    with open('HW5\X_test.csv', newline='',) as csvfile:
        rows = csv.reader(csvfile)
        x_test_buf = list(rows)
        for i in range(len(x_test_buf)):
            buf = []
            for j in range(len(x_test_buf[i])):
                buf.append(float(x_test_buf[i][j]))
            x_test_list.append(buf)
        # print(data_list)
        # x_test_buf = np.array(data_list)
        # print(x_train_buf.shape)
    with open('HW5\X_train.csv', newline='',) as csvfile:
        rows = csv.reader(csvfile)
        x_train_buf = list(rows)
        for i in range(len(x_train_buf)):
            buf = []
            for j in range(len(x_train_buf[i])):
                buf.append(float(x_train_buf[i][j]))
            x_train_list.append(buf)
        # x_train_buf = np.array(data_list)
    with open('HW5\Y_train.csv', newline='',) as csvfile:
        rows = csv.reader(csvfile)
        data_list = list(rows)
        y_train_buf = np.array(data_list)
        for i in range(len(y_train_buf)):
            y_train_buf1.append(int(y_train_buf[i]))
        # print(y_train_buf1)
    with open('HW5\Y_test.csv', newline='',) as csvfile:
        rows = csv.reader(csvfile)
        data_list = list(rows)
        y_test_buf = np.array(data_list)
        for i in range(len(y_test_buf)):
            y_test_buf1.append(int(y_test_buf[i]))
    x_train_list = np.array(x_train_list)
    x_test_list = np.array(x_test_list)
    y_train_buf1 = np.array(y_train_buf1)
    y_test_buf1 = np.array(y_test_buf1)
    return x_train_list, x_test_list, y_train_buf1, y_test_buf1


def compare_diff_performance(y_train, x_train, x_test, y_test):
    prob = svm_problem(y_train, x_train)
    # t - kernel_type l(0) p(1) r(2) / s - svm_type C-SVC(0)
    linear_param = svm_parameter('-t 0 -s 0 -q')
    polynomial_param = svm_parameter('-t 1 -s 0 -q')
    RBF_param = svm_parameter('-t 2 -s 0 -q')

    startTime = time.time()
    linear_m = svm_train(prob, linear_param)
    predict(linear_m, x_test, y_test, "linear")
    endTime = time.time()
    print(f'linear total time: {endTime - startTime}\n')

    startTime = time.time()
    polynomial_m = svm_train(prob, polynomial_param)
    predict(polynomial_m, x_test, y_test, "polynomial")
    endTime = time.time()
    print(f'polynomial total time: {endTime - startTime}\n')

    startTime = time.time()
    RBF_m = svm_train(prob, RBF_param)
    predict(RBF_m, x_test, y_test, "RBF")
    endTime = time.time()
    print(f'RBF total time: {endTime - startTime}\n')
    # return linear_m,polynomial_m,RBF_m


def predict(model, x_test, y_test, mode):
    if mode == "linear":
        print("linear kernel performance")
        svm_predict(y_test, x_test, model)
    if mode == "polynomial":
        print("polynomial kernel performance")
        svm_predict(y_test, x_test, model)
    if mode == "RBF":
        print("RBF kernel performance")
        svm_predict(y_test, x_test, model)


def compare_acc(best_parameter, best_accuracy, parameter, accuracy, best_kernel, kernel):
    if accuracy > best_accuracy:
        return parameter, accuracy, kernel
    return best_parameter, best_accuracy, best_kernel


def grid_search(x_train, x_test, y_train, y_test):
    best_option = []

    kernel = ["linear", "polynomial", "RBF"]
    # parameters
    costs = [0.1, 1, 10, 100]
    gammas = [1/784, 1, 0.1, 0.01]
    degrees = [0, 1, 2, 3]
    coef0s = [0, 1, 2, 3]

    prob = svm_problem(y_train, x_train)

    for i in range(len(kernel)):
        if kernel[i] == "linear":
            best_parameter = ''
            best_accuracy = 0.0
            best_kernel = ''
            print("-----linear------")
            for cost in costs:
                parameter = f'-t 0 -s 0 -c {cost} -v 5 -q'
                print(f"parameters: {parameter}")
                linear_param = svm_parameter(parameter)
                linear_m_acc = svm_train(prob, linear_param)
                # print(linear_m_acc)
                best_parameter, best_accuracy, best_kernel = compare_acc(
                    best_parameter, best_accuracy, parameter, linear_m_acc, best_kernel, kernel[i])
                # print("\n")
            buf = []
            buf.append(best_parameter)
            buf.append(best_accuracy)
            buf.append(best_kernel)
            best_option.append(buf)
        if kernel[i] == "polynomial":
            print("\n-----polynomial------")
            best_parameter = ''
            best_accuracy = 0.0
            best_kernel = ''
            for cost in costs:
                for gamma in gammas:
                    for degree in degrees:
                        for coef0 in coef0s:
                            parameter = f'-t 1 -s 0 -c {cost} -g {gamma} -d {degree} -r {coef0} -v 5 -q'
                            print(f"parameters: {parameter}")
                            polynomial_param = svm_parameter(parameter)
                            polynomial_m_acc = svm_train(
                                prob, polynomial_param)
                            # print(polynomial_m_acc)
                            best_parameter, best_accuracy, best_kernel = compare_acc(
                                best_parameter, best_accuracy, parameter, polynomial_m_acc, best_kernel, kernel[i])
                            # print("\n")
            buf = []
            buf.append(best_parameter)
            buf.append(best_accuracy)
            buf.append(best_kernel)
            best_option.append(buf)
        if kernel[i] == "RBF":
            print("\n-----radial basis function------")
            best_parameter = ''
            best_accuracy = 0.0
            best_kernel = ''
            for cost in costs:
                for gamma in gammas:
                    parameter = f'-t 2 -s 0 -c {cost} -g {gamma} -v 5 -q'
                    print(f"parameters: {parameter}")
                    RBF_param = svm_parameter(parameter)
                    RBF_m_acc = svm_train(prob, RBF_param)
                    # print(polynomial_m_acc)
                    best_parameter, best_accuracy, best_kernel = compare_acc(
                        best_parameter, best_accuracy, parameter, RBF_m_acc, best_kernel, kernel[i])
                    # print("\n")
            buf = []
            buf.append(best_parameter)
            buf.append(best_accuracy)
            buf.append(best_kernel)
            best_option.append(buf)

    print("\nthe best accuracy to each kernel to predict")
    print(best_option)
    for i in range(len(best_option)):
        best_option[i][0] = best_option[i][0].replace(" -v 5", "")
        model = svm_train(prob, best_option[i][0])
        predict(model, x_test, y_test, best_option[i][2])
        print(f"parameters: {best_option[i][0]}\n")


def rbf_kernel(x, y, gamma):
    return np.exp(-gamma * cdist(x, y, 'sqeuclidean'))


def linear_kernel(x, y):
    return x.dot(y.T)


def grid_search_rbf_and_linear(x_train, x_test, y_train, y_test):
    costs = [0.1, 1, 10, 100]
    gammas = [1/784, 1, 0.1, 0.01]
    best_parameter = ''
    best_accuracy = 0.0
    best_kernel = ''
    best_gamma = 0.0
    linear_k = linear_kernel(x_train, x_train)
    for cost in costs:
        for gamma in gammas:
            rbf_k = rbf_kernel(x_train, x_train, gamma)
            # linear_k = linear_kernel(x_train, y_train)
            X_kernel = np.hstack(
                (np.arange(1, 5001).reshape((-1, 1)), linear_k + rbf_k))
            parameter = f'-t 4 -c {cost} -v 5 -q'
            print(f"parameters: {parameter}")
            print(f"gamma: {gamma}")
            combine_param = svm_parameter(parameter)
            prob = svm_problem(y_train, X_kernel, True)
            combine_m_acc = svm_train(prob, combine_param)
            best_parameter, best_accuracy, best_kernel = compare_acc(
                best_parameter, best_accuracy, parameter, combine_m_acc, best_kernel, "rbf_and_linear")
            if best_accuracy == combine_m_acc:
                best_gamma = gamma
    # print(best_gamma)

    print("\n-----combine kernel predict part-----")
    best_parameter = best_parameter.replace(" -v 5", "")
    linear_k = linear_kernel(x_test, x_test)
    rbf_k = rbf_kernel(x_test, x_test, best_gamma)
    X_kernel = np.hstack(
        (np.arange(1, 2501).reshape((-1, 1)), linear_k + rbf_k))

    prob = svm_problem(y_test, X_kernel, True)
    model = svm_train(prob, best_parameter)
    svm_predict(y_test, X_kernel, model)
    print(f"parameters: {best_parameter}\ngamma: {best_gamma}")


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load()

    compare_diff_performance(y_train, x_train, x_test, y_test)

    print("----------------------------------------")

    grid_search(x_train, x_test, y_train, y_test)

    print("----------------------------------------")

    grid_search_rbf_and_linear(x_train, x_test, y_train, y_test)
