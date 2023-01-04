import numpy as np
import os,re
from PIL import Image
from scipy.spatial.distance import cdist
import random
import copy
from matplotlib import pyplot as plt
import imageio

def load(path):
    data = []
    label = []
    name=[]
    allFileList = os.listdir(path)
    for file in allFileList:           
        filename = f'{path}{file}'
        name.append(filename)
        image = Image.open(filename)
        data_buf = np.array(image)
        # print(data_buf)
        data_resize = np.zeros((21,15))
        for i in range(data_buf.shape[0]):
            for j in range(data_buf.shape[1]):
                data_resize[int(i/11)][int(j/13)] += data_buf[i][j]
        for i in range(data_resize.shape[0]):
            for j in range(data_resize.shape[1]):
                data_resize[i][j] = int(data_resize[i][j]/143)

        data_resize = data_resize.reshape(1,-1)
        data.append(data_resize[0])
        label.append(int(re.findall('(\d+)', filename)[0]))

    return  np.array(data),np.array(label),np.array(name)

def rbf_kernel(x, y, gamma):
    K = np.exp(-gamma * cdist(x.T, y.T, 'sqeuclidean'))
    # print(K.shape)
    N = np.ones((x.T.shape[0],x.T.shape[0]))/x.T.shape[0] #1N is NxN matrix with every element 1/N
    KC = K - N @ K - K @ N + N @ K @ N
    return KC

def linear_kernel(x, y):
    K = x.dot(y.T)
    N = np.ones((x.shape[0],x.shape[0]))/x.shape[0] #1N is NxN matrix with every element 1/N
    KC = K - N @ K - K @ N + N @ K @ N
    return KC

def polynomial_kernel(x, gamma, coef, degree):
    K = np.power(gamma * (x.T @ x) + coef, degree)
    N = np.ones((x.T.shape[0],x.T.shape[0]))/x.shape[0] #1N is NxN matrix with every element 1/N
    KC = K - N @ K - K @ N + N @ K @ N
    return KC

def PCA(train_data, train_label, test_data, test_label, kernel_type):
    mean = np.mean(train_data, axis=0)

    if kernel_type == "polynomial":
        gram = polynomial_kernel((train_data - mean)/np.std(train_data), 5, 10, 2)

    if kernel_type == "RBF":
        gram = rbf_kernel((train_data - mean)/np.std(train_data), (train_data - mean)/np.std(train_data), 0.01)

    if kernel_type == "simple":
        gram = (((train_data - mean)/np.std(train_data)).T @ (train_data - mean)/np.std(train_data)) 

    eigenvalue, eigenvector = np.linalg.eig(gram)
    targetIndex = np.argsort(-eigenvalue)
    eigenvalue = eigenvalue[targetIndex]
    eigenvector = eigenvector[:,targetIndex]
    eigenvector = eigenvector.real
    eigenvector = eigenvector[:,:25]

    for i in range(25):
        plt.figure("PCA 25 eigenfaces")
        plt.subplot(5, 5, i+1)
        plt.axis("off")
        plt.imshow(eigenvector[:,i].reshape(21,15), cmap="gray")   
    plt.show()
    reconstruct_num = np.random.choice(train_data.shape[0],10, replace=False)
    reconstruct = np.zeros((10, 315)) # 45045 = 231 * 195
    for i in range(10):
        reconstruct[i,:] = (train_data[reconstruct_num[i],:] - mean) @ eigenvector @ eigenvector.T + mean
    for i in range(10):
        plt.figure("PCA 10 recnstruction.")
        # Original image.
        plt.subplot(2 , 10, i + 1)
        plt.axis('off')
        plt.imshow(train_data[reconstruct_num[i], :].reshape((21, 15)), cmap='gray')

        # Reconstructed image.
        plt.subplot(2, 10, i + 11)
        plt.axis('off')
        plt.imshow(reconstruct[i, :].reshape((21, 15)), cmap='gray')
    plt.show()
    acc = prediction(train_data, train_label, test_data, test_label, eigenvector, mean)
    print(f"{kernel_type} PCA accuracy : {acc}")
    
    
def LDA(train_data, train_label, test_data, test_label,kernel_type):
    mean = np.mean(train_data, axis=0)
    training = copy.deepcopy(train_data)
    training = (train_data - mean)/np.std(train_data)
    mean_Standard = np.mean(training, axis=0)

    count = len(np.unique(train_label))
    SW = np.zeros((training.shape[1],training.shape[1]))
    SB = np.zeros((training.shape[1],training.shape[1]))
    
    for i in range(count): # 15
        Xi = training[np.where(train_label == i+1)[0], :]

        mj = np.mean(Xi, axis=0)

        if kernel_type == "simple":
            SW += (Xi - mj).T @ (Xi - mj)
            SB += 9 * (mj - mean_Standard).reshape(-1,1) @ (mj - mean_Standard).reshape(-1,1).T
        if kernel_type == "polynomial":
            SW += polynomial_kernel((Xi - mj), 5, 10, 2)
            SB += 9 * polynomial_kernel((mj - mean_Standard).reshape(-1,1).T, 5, 10, 2)
        if kernel_type == "RBF":
            SW += rbf_kernel((Xi - mj), (Xi - mj), 0.01)
            SB += 9 * rbf_kernel((mj - mean_Standard).reshape(-1,1).T, (mj - mean_Standard).reshape(-1,1).T, 0.01)

    gram = SB @ np.linalg.inv(SW) 
    
    eigenvalue, eigenvector = np.linalg.eig(gram)
    targetIndex = np.argsort(-eigenvalue)
    eigenvalue = eigenvalue[targetIndex]
    eigenvector = eigenvector[:,targetIndex]
    eigenvector = eigenvector.real
    eigenvector = eigenvector[:,:25]

    for i in range(25):
        plt.figure("LDA 25 eigenfaces")
        plt.subplot(5, 5, i+1)
        plt.axis("off")
        plt.imshow(eigenvector[:,i].reshape(21,15), cmap="gray")   
    plt.show()
    reconstruct_num = np.random.choice(training.shape[0], 10, replace=False)
    reconstruct = np.zeros((10, 315)) # 315 = 21 * 15

    for i in range(10):
        reconstruct[i,:] = ((training[reconstruct_num[i],:]) @ eigenvector @ eigenvector.T) * np.std(train_data) + mean

    for i in range(10):
        plt.figure("LDA 10 recnstruction.")
        # Original image.
        plt.subplot(2 , 10, i + 1)
        plt.axis('off')
        plt.imshow(train_data[reconstruct_num[i], :].reshape((21, 15)), cmap='gray')

        # Reconstructed image.
        plt.subplot(2, 10, i + 11)
        plt.axis('off')
        plt.imshow(reconstruct[i, :].reshape((21, 15)), cmap='gray')
    plt.show()
    acc = prediction(train_data, train_label, test_data, test_label, eigenvector, mean)
    print(f"{kernel_type} LDA accuracy : {acc}")
        

    


def prediction(train_data, train_label, test_data, test_label, eigenvector, mean):
    k = 3
    acc = 0

    train_proj = (train_data - mean) @ eigenvector
    test_proj = (test_data - mean) @ eigenvector
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
    
    kernel_mode = ["simple","polynomial","RBF"]
    
    for i in kernel_mode:
        print("PCA...")
        PCA(train_data, train_label, test_data, test_label,i)
        print("LDA...")
        LDA(train_data, train_label, test_data, test_label,i)

    