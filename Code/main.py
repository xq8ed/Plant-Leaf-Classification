""" main method to execute all functions."""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pandas
from module1 import preprocess, postprocess
from module2 import shape, margin

def save_image(path, img):
  if(cv.imwrite(path+'_bw.png', img)==True):
    print('Image written succesfully')
  else:
    print('Failed to write Image')

DATA_IN_PATH = "..\\Data\\RGB"
DATA_OUT_PATH = "..\\Extracted_data"
IMG_NAME_LIST = [f for dp, dn, filenames in os.walk(DATA_IN_PATH) for f in filenames if os.path.splitext(f)[1].lower() == '.jpg']

shape_vec, margin_vec = [], []
for IMG_NAME in IMG_NAME_LIST:
	IMG_PATH = os.path.join(DATA_IN_PATH, IMG_NAME)
	img = preprocess.prep(IMG_PATH)
	img = postprocess.postp(img)
	iname = os.path.splitext(IMG_NAME)[0]
	save_image(os.path.join(DATA_OUT_PATH, "images", iname), img)
	shape_vec.append(shape.shape_vector(img))
	margin_vec.append(margin.margin_vector(img))

df_shape = pandas.DataFrame(shape_vec, index= IMG_NAME_LIST)
df_shape.to_csv(os.path.join(DATA_OUT_PATH, "features", "shape.csv"), float_format='%.8f')
df_margin = pandas.DataFrame(margin_vec, index= IMG_NAME_LIST)
df_margin.to_csv(os.path.join(DATA_OUT_PATH, "features", "margin.csv"), float_format='%.8f')