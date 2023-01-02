import numpy as np
import os,re
from PIL import Image
from scipy.spatial.distance import cdist
import random
import copy
from matplotlib import pyplot as plt

def load(path):
    data = []
    label = []
    name=[]
    allFileList = os.listdir(path)
    for file in allFileList:           
        filename = f'{path}{file}'
        name.append(filename)
        image = Image.open(filename)
        data_buf = np.array(image).reshape(1,-1)
        data.append(data_buf[0])
        label.append(int(re.findall('(\d+)', filename)[0]))
    return  np.array(data),np.array(label),np.array(name)

def PCA(train_data, train_label, test_data, test_label):
    mean = np.mean(train_data, axis=0)
    cov = ((train_data - mean) @ (train_data - mean).T) / train_data.shape[0]
    
    # print(cov.shape)
    eigenvalue, eigenvector = np.linalg.eigh(cov)
    targetIndex = np.argsort(-eigenvalue)[:25]
    eigenvalue = eigenvalue[targetIndex]
    eigenvector = eigenvector[:,targetIndex]
    for i in range(eigenvector.shape[1]):
        eigenvector[:,i] /= np.linalg.norm(eigenvector[:, i])
    transform = (train_data - mean).T @ eigenvector
    for i in range(25):
        plt.figure("PCA 25 eigenfaces")
        plt.subplot(5, 5, i+1)
        plt.axis("off")
        plt.imshow(transform[:,i].reshape(231,195), cmap="gray")   
    # plt.show()
    reconstruct_num = np.random.choice(train_data.shape[0],10, replace=False)
    reconstruct = np.zeros((10, 45045)) # 45045 = 231 * 195
    for i in range(10):
        reconstruct[i,:] = (train_data[reconstruct_num[i],:] - mean) @ transform @ transform.T + mean
    for i in range(10):
        plt.figure("PCA 10 recnstruction.")
        # Original image.
        plt.subplot(2 , 10, i + 1)
        plt.axis('off')
        plt.imshow(train_data[reconstruct_num[i], :].reshape((231, 195)), cmap='gray')

        # Reconstructed image.
        plt.subplot(2, 10, i + 11)
        plt.axis('off')
        plt.imshow(reconstruct[i, :].reshape((231, 195)), cmap='gray')
    # plt.show()
    acc = prediction(train_data, train_label, test_data, test_label, transform, mean)
    print(f"simple PCA accuracy : {acc}")

def prediction(train_data, train_label, test_data, test_label, transform, mean):
    k = 3
    acc = 0
    train_proj = (train_data - mean) @ transform
    test_proj = (test_data - mean) @ transform
    distance = np.zeros(135)
    for testIndex, test in enumerate(test_proj):
        for trainIndex, train in enumerate(train_proj):
            distance[trainIndex] = np.linalg.norm(test - train)
        minDistances = np.argsort(distance)[:k]
        predict = np.argmax(np.bincount(train_label[minDistances]))
        if predict == test_label[testIndex]:
            acc += 1
    return acc/test_label.shape[0]

if __name__ == "__main__":
    print("loading...")
    train_data, train_label,train_name = load("./Training/")
    test_data, test_label,test_name = load("./Testing/")
    print("PCA...")
    PCA(train_data, train_label, test_data, test_label)