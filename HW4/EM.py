import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import numba as nb


def load():
    image_file = open('train-images.idx3-ubyte', 'rb')
    label_file = open('train-labels.idx1-ubyte', 'rb')
    magic_number = int.from_bytes(image_file.read(4), byteorder='big')
    number_of_images = int.from_bytes(image_file.read(4), byteorder='big')
    row = int.from_bytes(image_file.read(4), byteorder='big')
    col = int.from_bytes(image_file.read(4), byteorder='big')
    label_file.read(8)
    # print(magic_number, number_of_images, row, col)
    trainingLabel = np.zeros(number_of_images, dtype=int)  # 1維
    # print(trainingLabel)
    trainingData = np.zeros((number_of_images, row*col), dtype=int)  # 3維
    # print(int.from_bytes(label_file.read(1), byteorder='big'))
    for i in range(number_of_images):
        trainingLabel[i] = label_file.read(1)[0]
        for j in range(row):
            for k in range(col):
                trainingData[i][col*j + k] = image_file.read(1)[0]/128

    label_file.close()
    image_file.close()

    return number_of_images, trainingLabel, trainingData.astype(float)


def show_image(P, count, diff):
    result = ""
    for i in range(10):
        result += f"class {i}:\n"
        for row in range(28):
            for column in range(28):
                if (P[i][row*28+column] >= 0.5):
                    result += "1 "
                else:
                    result += "0 "
            result += "\n"
        result += "\n"
    result += f"No. of Iteration: {count}, Difference: {diff}\n\n"
    result += "------------------------------------------------------------\n\n"
    return result


def SetResult(P, i, num):
    result = ""
    result += f"labeled class {num}:\n"
    for row in range(28):
        for column in range(28):
            if (P[i][row*28+column] >= 0.5):
                result += "1 "
            else:
                result += "0 "
        result += "\n"
    result += "\n"
    return result


def EM(N, data):
    result = ""
    C = np.full((10), 0.1)
    P = np.random.rand(10, 784)
    W = np.zeros((N, 10))
    count = 0
    while (True):
        count += 1
        W = E_step(N, C, P, data)
        # print(W[0])
        C_NEW, P_NEW = M_step(N, W, data)
        diff_P = np.linalg.norm(P_NEW - P)
        diff_C = np.linalg.norm(C_NEW - C)
        C = C_NEW
        P = P_NEW
        result += show_image(P, count, diff_P)
        if (count >= 20 or (diff_P < 1e-2 and diff_C < 1e-2)):
            break

    return C, P, W, result, count


@njit()
def E_step(N, C, P, data):
    W = np.zeros((N, 10))
    for n in range(N):
        for i in range(10):
            W[n][i] = np.log(C[i])
            for k in range(784):
                if data[n][k] == 1:
                    W[n][i] = W[n][i] + np.log(P[i][k])
                else:
                    W[n][i] = W[n][i] + np.log(1-P[i][k])

        W[n] = np.exp(W[n] - max(W[n]))
        normalize = np.sum(W[n])
        if normalize != 0:
            W[n] /= normalize
    return W


def M_step(N, W, data):
    C_NEW = np.zeros(10)
    for i in range(10):
        C_NEW[i] = np.sum(W.T[i]) / N
    P_NEW = W.T.dot(data)
    for i in range(10):
        P_NEW[i] /= np.sum(W.T[i])

    return C_NEW, P_NEW


def inarray(table, value):
    for i in range(len(table)):
        if (table[i] == value):
            return i
    return -1


@njit()
def make_prediction(X, C, P, label):
    predict_gt = np.zeros((10, 10))
    pdistribution = np.zeros(10)
    for i in range(60000):
        for k in range(10):
            pdistribution[k] = C[k, 0]
            for d in range(784):
                if X[i][d] == 1:
                    pdistribution[k] *= P[k][d]
                else:
                    pdistribution[k] *= (1 - P[k][d])
        predict = np.argmax(pdistribution)
        predict_gt[predict, label[i]] += 1
    return predict_gt


def prediction(N, C, P, W, label, data):
    result = ""
    predict = np.zeros(N, dtype=int)
    table = np.full((10), -1)
    for i in range(N):
        predict[i] = np.argmax(W[i])
    index = 0
    while np.any(table < 0):
        if (table[label[index]] == -1) & (inarray(table, predict[index]) == -1):
            table[label[index]] = predict[index]
        index += 1
        if (index >= 60000):
            for i in range(10):
                if (inarray(table, i) == -1):
                    table[inarray(table, -1)] = i
            break
    result = "labeled "
    for i in range(10):
        result += SetResult(P, table[i], i)

    confusion_matrix = np.zeros((10, 2, 2), dtype=int)
    for n in range(N):
        T = inarray(table, predict[n])
        if (label[n] == T):
            confusion_matrix[label[n]][0][0] += 1
            for i in range(10):
                if (i != label[n]):
                    confusion_matrix[i][1][1] += 1
        if (label[n] != T):
            confusion_matrix[label[n]][0][1] += 1
            for i in range(10):
                if (i != label[n]) & (T == i):
                    confusion_matrix[i][1][0] += 1
                elif (i != label[n]) & (T != i):
                    confusion_matrix[i][1][1] += 1

    result += SetCMResult(confusion_matrix, count, N)
    return result


def final_imagination(p, predict_gt_relation):
    result = ""
    im = (p >= 0.5)*1
    for c in range(10):
        for k in range(10):
            if predict_gt_relation[k] == c:
                choose = k
        result += f"labeled class {c}:\n"
        for row in range(28):
            for col in range(28):
                result += f"{im[choose][row*28+col]} "
            result += "\n"
        result += "\n"
    return result


def SetCMResult(confusion_matrix, count, N):
    result = ""
    currect = 0
    for i in range(10):
        currect += confusion_matrix[i][0][0]
        result += f"Confusion Matrix {i}:\n"
        result += f"\t\t\t\tPredict number {i} Predict not number {i}\n"
        # result += "{0:<15} {1:^16} {2:^20}\n".format(f"Is number {i}", f"{confusion_matrix[i][0][0]}", f"{confusion_matrix[i][0][1]}")
        result += f"Is number {i}\t\t\t{confusion_matrix[i][0][0]}\t\t\t{confusion_matrix[i][0][1]}\n"
        # result += "{0:<15} {1:^16} {2:^20}\n".format(f"Isn't number {i}", f"{confusion_matrix[i][1][0]}", f"{confusion_matrix[i][1][1]}")
        result += f"Isn't number {i}\t\t\t{confusion_matrix[i][1][0]}\t\t\t{confusion_matrix[i][1][1]}\n"
        result += "\n"
        result += "Sensitivity (Successfully predict number {0}):     {1}\n".format(
            f"{i}", f"{confusion_matrix[i][0][0]/(confusion_matrix[i][0][0]+confusion_matrix[i][0][1])}")
        result += "Speciticity (Successfully predict not number {0}): {1}\n".format(
            f"{i}", f"{confusion_matrix[i][1][0]/(confusion_matrix[i][1][0]+confusion_matrix[i][1][1])}")
        result += "------------------------------------------------------------\n\n"

    result += f"Total iteration to converge: {count}\n"
    result += f"Total error rate: {1 - currect/N}"
    return result


if __name__ == "__main__":
    print("load ...")
    N, label, data = load()  # 60000, 60000, 60000*784 -> 0 or 1
    print("EM ...")
    C, P, W, result, count = EM(N, data)

    print("prediction ...")
    result += prediction(N, C, P, W, label, data)
    resultFile = open(f"result.txt", 'w')
    resultFile.write(result)
