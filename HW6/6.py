import numpy as np
import random
from scipy.spatial.distance import cdist
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
import os
import cv2


def load():
    # img1 = Image.open("Machine_Learning\HW6\image1.png")
    # img2 = Image.open("Machine_Learning\HW6\image2.png")
    imgcv2_1 = cv2.imread('Machine_Learning\HW6\image1.png')
    # print(imgcv2_1.shape)
    data_Color = imgcv2_1.reshape(-1, imgcv2_1.shape[2])
    data_Spatial = np.array([(i, j) for i in range(
        imgcv2_1.shape[0])for j in range(imgcv2_1.shape[1])])
    return data_Color, data_Spatial


def kernel_k_means(data_Color, data_Spatial):
    s = 0.001
    c = 0.001
    gram = np.exp(-s * cdist(data_Spatial, data_Spatial, 'sqeuclidean'))
    gram *= np.exp(-c * cdist(data_Color, data_Color, 'sqeuclidean'))
    return gram


if __name__ == "__main__":
    data_Color, data_Spatial = load()
    Gram = kernel_k_means(data_Color, data_Spatial)
    print(Gram.shape)
    # data1, data2 = load()
    # print(data1)
    # data1, data2 = load()
