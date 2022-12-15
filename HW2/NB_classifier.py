import numpy as np
import struct
import math
import numba as nb
# 讀黨


@nb.njit()
def Discrete(data, labels):
    prior = np.zeros(10)
    likelihood = np.ones((10, 28, 28, 32))
    for i in range(len(labels)):
        # print(labels[i])
        prior[labels[i]] += 1  # 看number = 0-9 各自有幾個
        for j in range(28):  # row
            for k in range(28):  # col
                # 0-255除以8來變成32bins
                likelihood[labels[i]][j][k][int(data[i][j][k]/8)] += 1
    for i in range(10):
        for j in range(28):
            for k in range(28):
                for l in range(32):
                    likelihood[i][j][k][l] = likelihood[i][j][k][l] / prior[i]
    for i in range(10):
        prior[i] /= 60000  # 計算出縣0-9的機率
    return prior, likelihood


@nb.njit()
def Continue(data, labels):
    prior = np.zeros(10)
    Gaussian = np.zeros((10, 28, 28, 2))  # 2 means mean and variance

    for i in range(len(labels)):
        label = labels[i]
        for j in range(28):
            for k in range(28):
                Gaussian[label][j][k][0] = (
                    prior[label]/(prior[label]+1)) * Gaussian[label][j][k][0] + data[i][j][k] / (prior[label]+1)
                Gaussian[label][j][k][1] = (
                    prior[label]/(prior[label]+1)) * Gaussian[label][j][k][1] + (data[i][j][k]**2) / (prior[label]+1)
        prior[label] += 1

    for i in range(10):
        for j in range(28):
            for k in range(28):
                Gaussian[i][j][k][1] -= Gaussian[i][j][k][0]**2

    return prior/60000, Gaussian


@nb.njit()
def test_discrete(labels, data, prior, likelihood):
    # print("a")
    err = 0.0
    posterior = np.zeros((len(labels), 10))
    predictions = np.zeros(len(labels))
    answers = np.zeros(len(labels))
    for i in range(len(labels)):
        # Calculate posterior
        for j in range(10):
            posterior[i] += np.log10(prior[j])
            for k in range(28):
                for l in range(28):
                    posterior[i][j] += np.log10(
                        likelihood[j][k][l][int(data[i][k][l]/8)])

        predictions[i] = np.argmax(posterior[i])
        answers[i] = labels[i]
        if predictions[i] != answers[i]:
            err += 1

    return posterior, predictions, answers, err/len(labels)


@nb.njit()
def test_continuous(labels, data, prior, likelihood):
    err = 0.0
    posterior = np.zeros((len(labels), 10))
    predictions = np.zeros(len(labels))
    answers = np.zeros(len(labels))
    for i in range(len(labels)):
        # Calculate posterior
        for j in range(10):
            posterior[i] += np.log10(prior[j])
            for k in range(28):
                for l in range(28):
                    mean = likelihood[j][k][l][0]
                    variance = likelihood[j][k][l][1]
                    if variance != 0:
                        # posterior[i][j] += math.log10(((math.exp(1))**((data[i][j][k] - mean) ** 2)/(-2)*(
                        #     variance**2))/math.sqrt(2*math.pi * (variance ** 2)))
                        posterior[i][j] += -0.5 * math.log10(2 * math.pi * variance) - math.log10(
                            math.exp(1)) * ((data[i][k][l] - mean) ** 2) / (2 * variance)

        predictions[i] = np.argmax(posterior[i])
        answers[i] = labels[i]
        if predictions[i] != answers[i]:
            err += 1

    return posterior, predictions, answers, err/len(labels)


def printResult_continuous(likelihood, posterior, predictions, answers, err):
    result = ""

    for image_index in range(len(predections)):
        result += "Posterior (in log scale):\n"
        for label in range(10):
            result += f"{label}: {posterior[image_index][label]/np.sum(posterior[image_index])}\n"
        result += f"Prediction: {predictions[image_index]}, Ans: {answers[image_index]}\n\n"

    # Print Bayesian classifier
    result += "Imagination of numbers in Bayesian classifier:"
    for label in range(10):
        result += f"\n{label}:\n"
        for i in range(28):
            for j in range(28):
                # The MLE of likelihood is mean
                classifier_value = likelihood[label][i][j][0]
                result += f"{int(classifier_value/128)} "
            result += "\n\n"

    result += f"Error rate: {err}"

    return result


def printResult_discrete(likelihood, posterior, predictions, answers, err):
    result = ""

    for image_index in range(len(predections)):
        result += "Posterior (in log scale):\n"
        for label in range(10):
            result += f"{label}: {posterior[image_index][label]/np.sum(posterior[image_index])}\n"
        result += f"Prediction: {predictions[image_index]}, Ans: {answers[image_index]}\n\n"

    # Print Bayesian classifier
    result += "Imagination of numbers in Bayesian classifier:"
    for label in range(10):
        result += f"\n{label}:\n"
        for i in range(28):
            for j in range(28):
                classifier_value = np.argmax(likelihood[label][i][j])
                result += f"{int(classifier_value/16)} "
            result += "\n\n"

    result += f"Error rate: {err}"

    return result


image_file = open('train-images.idx3-ubyte', 'rb')
label_file = open('train-labels.idx1-ubyte', 'rb')
magic_number = int.from_bytes(image_file.read(4), byteorder='big')
number_of_images = int.from_bytes(image_file.read(4), byteorder='big')
row = int.from_bytes(image_file.read(4), byteorder='big')
col = int.from_bytes(image_file.read(4), byteorder='big')
label_file.read(8)
# print(magic_number, number_of_images, row, col)
trainingLabel = np.zeros(number_of_images, dtype=int)
# print(trainingLabel)
trainingData = np.zeros((number_of_images, row, col), dtype=int)
# print(int.from_bytes(label_file.read(1), byteorder='big'))
for i in range(number_of_images):
    trainingLabel[i] = label_file.read(1)[0]
    for j in range(row):
        for k in range(col):
            trainingData[i][j][k] = image_file.read(1)[0]
# print(trainingLabel)
# print(trainingData[0])
label_file.close()
image_file.close()
image_file = open('t10k-images.idx3-ubyte', 'rb')
label_file = open('t10k-labels.idx1-ubyte', 'rb')
label_file.read(8)
magic_number = int.from_bytes(image_file.read(4), byteorder='big')
number_of_images = int.from_bytes(image_file.read(4), byteorder='big')
row = int.from_bytes(image_file.read(4), byteorder='big')
col = int.from_bytes(image_file.read(4), byteorder='big')
# print(magic_number, number_of_images, row, col)
testLabel = np.zeros(number_of_images, dtype=int)
# print(testLabel)
testData = np.zeros((number_of_images, row, col), dtype=int)
for i in range(number_of_images):
    testLabel[i] = label_file.read(1)[0]
    for j in range(row):
        for k in range(col):
            testData[i][j][k] = image_file.read(1)[0]
label_file.close()
image_file.close()


prior, likelihood = Discrete(trainingData, trainingLabel)
posterior, predections, answers, err = test_discrete(
    testLabel, testData, prior, likelihood)
resultFile = open("result_discrete.txt", 'w')
resultFile.write(printResult_discrete(
    likelihood, posterior, predections, answers, err))
# result = printResult(likelihood, posterior, predections, answers, err)
# print(result)
prior, likelihood = Continue(trainingData, trainingLabel)
posterior, predections, answers, err = test_continuous(
    testLabel, testData, prior, likelihood)
resultFile = open("result_continuous.txt", 'w')
resultFile.write(printResult_continuous(
    likelihood, posterior, predections, answers, err))
