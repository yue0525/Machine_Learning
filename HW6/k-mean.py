import numpy as np
import random
from scipy.spatial.distance import cdist
from PIL import Image
import copy


def load(image_num):
    str_image = f"image{image_num}.png"
    img1 = Image.open(str_image)
    data = np.array(img1)
    data_size = data.shape[0]
    data_Color = data.reshape(-1, data.shape[2])
    # print(data_Color.shape)
    data_Spatial = np.array([(i, j) for i in range(data.shape[0])for j in range(data.shape[1])])
    # print(data_Spatial.shape)
    return data_Color, data_Spatial, data_size
    

def kernel_k_means(data_Color, data_Spatial):
    s = 0.001
    c = 0.001
    gram = np.exp(-s * cdist(data_Spatial, data_Spatial, 'sqeuclidean'))
    gram *= np.exp(-c * cdist(data_Color, data_Color, 'sqeuclidean'))
    return gram


def k_mean(Gram,k_num,kmean_kmeanplusplus):
    k = k_num
    mode = kmean_kmeanplusplus
    mean = np.zeros((k,Gram.shape[0]))
    # mode = 0 k-mean
    if mode == 0:
        # select k centers of random
        center = random.sample(range(0, 10000), k)
        center = np.array(center)   
        # set the center to the mean
        for i in range(k):
            mean[i] = Gram[center[i]]
    # print(mean)
    # mode = 1 k-mean++
    if mode == 1:
        random_num = random.sample(range(0, 10000),1)
        mean[0] = Gram[random_num]
        for mean_id in range(1,k):

            dis = []
            for i in range(Gram.shape[0]):
                dis.append(np.linalg.norm(Gram[i] - mean[mean_id - 1]) ** 2)
            dis_sum = np.sum(dis)
            p_dis = []
            for i in range(len(dis)):
                p_dis.append(dis[i]/dis_sum)
            sum_p = np.cumsum(p_dis)
            random_pro = np.random.uniform(0,1)
            mean_num = 0
            for i in range(len(sum_p)-1):
                if random_pro < sum_p[0]:
                    break
                elif random_pro > sum_p[i] and random_pro < sum_p[i + 1]:
                    mean_num = i
                    break
                if i == (len(sum_p)-2) and random_pro > sum_p[i + 1]:
                    mean_num = i + 1
            mean[mean_id] = Gram[mean_num]
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

def make_gif(gif_pic,data_size,count,k,kmean_kmeanplusplus,image_num):
    mode = kmean_kmeanplusplus
    images = []
    color = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)] # r g b y
    width = data_size
    for i in range(count):
        images.append(Image.new('RGB', (width, width)))
        for x in range(width):
            for y in range(width):
                images[i].putpixel((y,x),color[gif_pic[i][x * width + y][0]])
    if mode == 0:
        images[0].save(f'./image{image_num}_k_mean_pic/kmeans_k-{k}.gif', format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)
        images[count-1].save(f'./image{image_num}_k_mean_pic/kmeans_k-{k}.png')
    if mode == 1:
        images[0].save(f'./image{image_num}_k_mean_pic/k-mean++_k-{k}.gif', format='GIF', append_images=images[1:], save_all=True, duration=100, loop=0)
        images[count-1].save(f'./image{image_num}_k_mean_pic/k-mean++_k-{k}.png')
    return
    

if __name__ == "__main__":    
    k_num = int(input("k = "))
    kmean_kmeanplusplus = int(input("using k-mean(0) or k-mean++(1) : "))
    image_num = int(input("which image (1) or (2) : "))
    print("loading...")

    data_Color, data_Spatial, data_size = load(image_num)

    print("get kernel...")

    Gram = kernel_k_means(data_Color, data_Spatial)

    print("k-means...")

    gif_pic ,count = k_mean(Gram,k_num,kmean_kmeanplusplus)

    print("making GIF...")

    make_gif(gif_pic,data_size,count,k_num,kmean_kmeanplusplus,image_num)
