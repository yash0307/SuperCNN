from math import *
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
from skimage import io, color

def generateCentroid(m, n, K):
    # m: number of rows
    # n: number of columns
    # K: number of superpixels required
    S = int(np.floor(sqrt(m * n / K)))  # stride step; ensures similarly sized superpixels
    # Initialize the K centroids, picked randomly from the SxS window
    count = 0
    centroid = np.empty([K, 2])
    for i in range(0, m, S):
        for j in range(0, n, S):
            begin_y = i
            end_y = i + S
            begin_x = j
            end_x = j + S
            if end_x > m:
                end_x = m
            if end_y > n:
                end_y = n
            centroid_x = np.random.choice(np.arange(begin_x, end_x))
            centroid_y = np.random.choice(np.arange(begin_y, end_y))
            print(count)
            centroid[count, :] = np.array([centroid_x, centroid_y])
            count = count + 1
    return centroid


# Initialize
im = imread("lena.png")
im = imresize(im, (500, 500, 3))
lab = color.rgb2lab(im)
m = np.size(lab, 0)
n = np.size(lab, 1)
# K = int(input("what should be the number of superpixels \n"))
K = 400 # Needs to be a perfect square
centroid = generateCentroid(m,n,K)
# Adjust the centroid initializations so the the point does not lie on gradient



plt.scatter(centroid[:, 0], centroid[:, 1], c='r', s=40)
plt.imshow(np.uint8(lab))
plt.show()
# lab = color.rgb2lab(im)
# np.size(lab, 1)
#
#
# print(np.size(lab), 1)
# print(S)
# print(np.shape(lab))
