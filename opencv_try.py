import json
import numpy as np
import cv2
from data import Images
from PIL import Image, ImageOps
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import random


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# cif = x_train[:200]
# print(cif.shape)
#
# np.save('cifar_10.npy', cif)

# with open('labels.json', 'r') as file:
#     f = json.load(file)
#
# c = 0
# for i in f['random']:
#     if i == f['mix'][i]:
#         c += 1
#
# print(c)

# number = 61
#
# arr = np.load(NPY_PATH)
# name = 'image_with_number_example_%s.png' % number
# img = cv2.imwrite(name, arr[number])
# print(img)
#
#C:\Users\oknav\Desktop\Csillu\egyi\szakdoga\autoencoders\clustering\cifar

# img = Images('C:/Users/oknav/Desktop/Csillu/egyi/szakdoga/autoencoders/clustering/X.npy')
# size = 256, 1024
#
# npy = np.load('C:/Users/oknav/Desktop/Csillu/egyi/szakdoga/autoencoders/clustering/X.npy')
# kezek = list()

# for i in range(200):
#     num = random.randint(0, npy.shape[0] - 1)
#     im = (npy[num]*255)-255
#     im = im*(-1)
#     print(np.amax(im))
#     orig = Image.fromarray(im)
#     plt.imshow(im)
#     # orig = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#     #
#     resized = orig.resize(size)
#     resized = np.array(resized)
#     resized = np.expand_dims(resized, axis=3)
#     img_path = 'C:/Users/oknav/Desktop/Csillu/egyi/szakdoga/autoencoders/clustering/fashion/image_%s.png' % i
#     # plt.savefig(img_path)
#     # img = Image.open('C:/Users/oknav/Desktop/Csillu/egyi/szakdoga/autoencoders/clustering/homogene/image_46.png').convert('RGB')
#     save = cv2.imwrite(img_path, resized)
#     kezek.append(resized)
#     print(save)
#
# kezek = np.array(kezek)
# np.save('C:/Users/oknav/Desktop/Csillu/egyi/szakdoga/autoencoders/clustering/fashion/kezek.npy', kezek)
# print(kezek.shape)
#
# img_path = 'C:/Users/oknav/Desktop/Csillu/egyi/szakdoga/autoencoders/clustering/fashion/kezek.npy'
# #
# # b = np.load(img_path)
# # num = random.randint(0, b.shape[0])
# # b = b[:, :, :, 0]
# # plt.imshow(b[num])
# # plt.imsave('test_im.png', b[num])

mix = np.load('C:/Users/oknav/Desktop/Csillu/egyi/szakdoga/autoencoders/clustering/mix_test.npy')

for i, im in enumerate(mix):
    im = im[:,:,0]
    plt.imsave('mix_%s' % i, im)
