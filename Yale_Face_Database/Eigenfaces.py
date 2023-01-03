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

def rbf_kernel(x, y, gamma):
    K = np.exp(-gamma * cdist(x, y, 'sqeuclidean'))
    N = np.ones((x.shape[0],x.shape[0]))/x.shape[0] #1N is NxN matrix with every element 1/N
    KC = K - N @ K - K @ N + N @ K @ N
    return KC

def linear_kernel(x, y):
    K=  x.dot(y.T)
    N = np.ones((x.shape[0],x.shape[0]))/x.shape[0] #1N is NxN matrix with every element 1/N
    KC = K - N @ K - K @ N + N @ K @ N
    return KC

def polynomial_kernel(x, gamma, coef, degree):
    K = np.power(gamma * (x @ x.T) + coef, degree)
    N = np.ones((x.shape[0],x.shape[0]))/x.shape[0] #1N is NxN matrix with every element 1/N
    KC = K - N @ K - K @ N + N @ K @ N
    return KC
def PCA(train_data, train_label, test_data, test_label, kernel_type):
    mean = np.mean(train_data, axis=0)

    if kernel_type == "linear":
        gram = linear_kernel(train_data,train_data)

    if kernel_type == "RBF":
        gram = rbf_kernel(train_data, train_data, 0.01)

    if kernel_type == "simple":
        gram = ((train_data - mean) @ (train_data - mean).T) 

    eigenvalue, eigenvector = np.linalg.eig(gram)
    targetIndex = np.argsort(-eigenvalue)
    eigenvalue = eigenvalue[targetIndex]
    eigenvector = eigenvector[:,targetIndex]
    for i in range(len(eigenvalue)):
        if (eigenvalue[i] <= 0):
            eigenvalue = eigenvalue[:i].real
            eigenvector = eigenvector[:, :i].real
            break
    eigenvector = eigenvector[:,:25]
    for i in range(eigenvector.shape[1]):
        eigenvector[:,i] /= np.linalg.norm(eigenvector[:, i])

    transform = (train_data - mean).T @ eigenvector

    z = transform.T @ (train_data - mean).T

    for i in range(25):
        plt.figure("PCA 25 eigenfaces")
        plt.subplot(5, 5, i+1)
        plt.axis("off")
        plt.imshow(transform[:,i].reshape(231,195), cmap="gray")   
    plt.show()
    reconstruct_num = np.random.choice(train_data.shape[0],10, replace=False)
    reconstruct = np.zeros((10, 45045)) # 45045 = 231 * 195
    for i in range(10):
        reconstruct[i,:] = (train_data[reconstruct_num[i],:] - mean) @ transform @ transform.T + mean
        # reconstruct[i,:] = train_data[reconstruct_num[i],:] @ transform @ transform.T 
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
    plt.show()
    acc = prediction(train_data, train_label, test_data, test_label, transform, mean)
    print(f"{kernel_type} PCA accuracy : {acc}")

    
    
def LDA(train_data, train_label, test_data, test_label,kernel_type):
    mean = np.mean(train_data, axis=0)
    if kernel_type == "linear":
        gram = linear_kernel(train_data,train_data)

    if kernel_type == "RBF":
        gram = rbf_kernel(train_data, train_data, 0.01)

    if kernel_type == "polynomial":
        gram = polynomial_kernel(train_data, 5, 10, 2)

    if kernel_type == "simple":
        gram = ((train_data - mean) @ (train_data - mean).T) 

    eigenvalue, eigenvector = np.linalg.eig(gram)
    targetIndex = np.argsort(-eigenvalue)
    eigenvalue = eigenvalue[targetIndex]
    eigenvector = eigenvector[:,targetIndex]
    for i in range(eigenvector.shape[1]):
        eigenvector[:,i] /= np.linalg.norm(eigenvector[:, i])
    transform = (train_data - mean).T @ eigenvector

    z = transform.T @ (train_data - mean).T

    n = z.shape[0]
    SW = np.zeros((n, n))
    mean_z = np.mean(z, axis=1)
    for i in range(15):
        SW += z[:, i*9:i*9+9] @ z[:, i*9:i*9+9].T

    SB = np.zeros((n, n))
    for i in range(15):
        class_mean = np.mean(z[:, i*9:i*9+9], axis=1).T
        SB += 9 * (class_mean - mean_z) @ (class_mean - mean_z).T
    gram = np.linalg.inv(SW) @ SB
    eigenvalue, eigenvector = np.linalg.eig(gram)

    targetIndex = np.argsort(-eigenvalue)
    eigenvalue = eigenvalue[targetIndex]
    eigenvector = eigenvector[:,targetIndex]
    for i in range(len(eigenvalue)):
        if (eigenvalue[i] <= 0):
            eigenvalue = eigenvalue[:i].real
            eigenvector = eigenvector[:, :i].real
            break
    eigenvector = eigenvector[:,:25]

    transform = (train_data - mean).T @ eigenvector
    z = transform.T @ (train_data - mean).T
    for i in range(25):
        plt.figure("LDA 25 eigenfaces")
        plt.subplot(5, 5, i+1)
        plt.axis("off")
        plt.imshow(transform[:,i].reshape(231,195), cmap="gray")   
    plt.show()
    reconstruct_num = np.random.choice(train_data.shape[0], 10, replace=False)
    reconstruct = np.zeros((10, 45045)) # 45045 = 231 * 195
    for i in range(10):
        reconstruct[i,:] = (train_data[reconstruct_num[i],:] - mean) @ transform @ transform.T + mean
        # reconstruct[i,:] = train_data[reconstruct_num[i],:] @ transform @ transform.T 
    for i in range(10):
        plt.figure("LDA 10 recnstruction.")
        # Original image.
        plt.subplot(2 , 10, i + 1)
        plt.axis('off')
        plt.imshow(train_data[reconstruct_num[i], :].reshape((231, 195)), cmap='gray')

        # Reconstructed image.
        plt.subplot(2, 10, i + 11)
        plt.axis('off')
        plt.imshow(reconstruct[i, :].reshape((231, 195)), cmap='gray')
    plt.show()
    acc = prediction(train_data, train_label, test_data, test_label, transform, mean)
    print(f"{kernel_type} LDA accuracy : {acc}")
        

    


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
    train_data, train_label,train_name = load("./Yale_Face_Database/Training/")
    test_data, test_label,test_name = load("./Yale_Face_Database/Testing/")
    
    kernel_mode = ["simple","linear","polynomial","RBF"]
    
    for i in kernel_mode:
        # print("PCA...")
        # PCA(train_data, train_label, test_data, test_label,i)
        # print(transform.shape)
        # print(z.shape)
        print("LDA...")
        LDA(train_data, train_label, test_data, test_label,i)
        

    