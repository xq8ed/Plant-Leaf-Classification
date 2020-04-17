# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:09:40 2020

@author: Ganesh
"""
import matplotlib.pyplot as plt
import preprocess
import postprocess

path = 'D:\Engineering\BE\Project\Datasets\Samples\RGB\\'
filename ='leaf_11'
ext = '.JPG'

img = preprocess.prep(path, filename, ext)
plt.imshow(img, cmap = 'gray')
plt.show()

img2 = postprocess.postp(img)
plt.imshow(img2, cmap = 'gray')
plt.show()

