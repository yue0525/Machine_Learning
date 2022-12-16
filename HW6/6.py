import numpy as np
import random
from scipy.spatial.distance import cdist
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageColor
from PIL import Image, ImageDraw
import os
# import cv2
import copy


def load():
    img1 = Image.open("image1.png")
    data = np.array(img1)
    # print(data.shape) 
    data_size = data.shape[0]
    data_Color = data.reshape(-1, data.shape[2])
    data_Spatial = np.array([(i, j) for i in range(data.shape[0])for j in range(data.shape[1])])
    # print(data_Spatial[0])
    return data_Color, data_Spatial, data_size
    

def kernel_k_means(data_Color, data_Spatial):
    s = 0.001
    c = 0.001
    gram = np.exp(-s * cdist(data_Spatial, data_Spatial, 'sqeuclidean'))
    gram *= np.exp(-c * cdist(data_Color, data_Color, 'sqeuclidean'))
    return gram


def k_mean(Gram):
    k = 2
    # select k centers of random
    center = random.sample(range(0, 10000), k)
    center = np.array(center)
    mean = np.zeros((k,Gram.shape[0]))
    # set the center to the mean
    for i in range(k):
        mean[i] = Gram[center[i]]
    # print(mean)
    gif_pic = []
    count = 0
    while(True):
        count += 1
        # E-step classify all samples according to closet
        cluster = [] # select which cluster is this point
        for i in range(Gram.shape[0]):
            #buffer is storing the kth euclidean norm
            buffer = []
            for j in range(k):
                buffer.append(np.linalg.norm(Gram[i] - mean[j]))
            cluster.append(np.argmin(buffer))

        cluster = np.array(cluster).reshape(-1,1)  
        pre_mean = copy.deepcopy(mean)

        # M-step re-compute as the mean μk of the points in cluster Ck
        for i in range(k):
            buffer = []
            for j in range(cluster.shape[0]):
                if cluster[j][0] == i:
                    buffer.append(Gram[j])
            mean[i] = np.mean(buffer, axis=0)

        # store the each cluster to gif_pic for making the .gif
        gif_pic.append(cluster)

        if np.linalg.norm(mean - pre_mean) < 1e-5:
            print(count)
            gif_pic = np.array(gif_pic)
            break
    
    return gif_pic, count

def make_gif(gif_pic,data_size,count):
    images = []
    color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)] # r g b y
    width = data_size
    for i in range(count):
        images.append(Image.new('RGB', (width, width)))
        for x in range(width):
            for y in range(width):
                images[i].putpixel((x,y),color[gif_pic[i][x * data_size + y][0]])

    images[0].save('kmeans.gif', format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)
    return

if __name__ == "__main__":
    data_Color, data_Spatial, data_size= load()
    # print(data_size)
    Gram = kernel_k_means(data_Color, data_Spatial)

    gif_pic ,count= k_mean(Gram)

    # make_gif(gif_pic,data_size,count)
