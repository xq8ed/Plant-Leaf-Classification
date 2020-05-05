# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:09:40 2020

@author: Ganesh
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from module1 import preprocess, postprocess
from module2 import feature_extract as fe

def save_image(path, filename, img):
  if(cv.imwrite(path+filename+'_bw.png', img)==True):
    print('Image written succesfully')
  else:
    print('Failed to write Image')

pathRGB = '../Data/Samples/RGB/'
filename ='leaf_11'
ext = '.JPG'

img = preprocess.prep(pathRGB, filename, ext)
fig = plt.figure(figsize=(18,12))
plt.subplot(221)
plt.imshow(img, cmap = 'gray')

img2 = postprocess.postp(img)
save_image('../Data/Samples/extracted/', filename, img2)
plt.subplot(222)
plt.imshow(img2, cmap = 'gray')

norm_shape_vec, interp_shape_vec = fe.feat_ext(img2)
x = np.arange(1, len(norm_shape_vec)+1)
y = norm_shape_vec
plt.subplot(223)
plt.title('Normalized Signature ('+str(len(norm_shape_vec))+' points)', size=18)
plt.plot(x, y)
plt.grid()

x = np.arange(1, len(interp_shape_vec)+1)
y = interp_shape_vec
plt.subplot(224)
plt.title('Interpolated Signature ('+str(len(interp_shape_vec))+' points)', size=18)
plt.plot(x, y)
plt.grid()
plt.show()