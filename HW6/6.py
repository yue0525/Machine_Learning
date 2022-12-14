import numpy as np
import random
from scipy.spatial.distance import cdist
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
import os


def load():
    img1 = Image.open("Machine_Learning\HW6\image1.png")
    img2 = Image.open("Machine_Learning\HW6\image1.png")
    data1 = np.array(img1)
    data2 = np.array(img2)
    # print(data1.shape)
    dataC = data1.reshape((data1.shape[0]*data1.shape[1], data1.shape[2]))
    # spatial data: coordinate for each pixel
    dataS = np.array([(i, j) for i in range(data1.shape[0])
                     for j in range(data1.shape[1])])
    print(img1.size)
    return data1, data2


def kernel_k_means():
    return


if __name__ == "__main__":
    data1, data2 = load()
    data1, data2 = load()
