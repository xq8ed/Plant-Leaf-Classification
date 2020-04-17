# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:45:17 2020

@author: Ganesh
"""
import numpy as np
import cv2 as cv
from math import sqrt
import scipy.ndimage as ndi
from shapely.geometry import LineString

from postprocess import get_contour


def get_center(contour):
  """ Returns centroid coords calculated with cv moments """
  M = cv.moments(contour)
  cmx = int(M['m10']/M['m00'])
  cmy = int(M['m01']/M['m00'])
  return cmx, cmy

def shift_to_origin(contour):
  """ Takes coords and shifts to origin """
  cx, cy = get_center(contour)
  contour[::,0] -= cx # deamon X
  contour[::,1] -= cy # deamon Y
  return contour

def extract_shape(img):
  """
  Expects prepared image, returns leaf shape in img format.
  The strength of smoothing had to be dynamically set
  in order to get consistent results for different sizes.
  """
  size = int(np.count_nonzero(img)/1000)
  brush = int(5*size/size**0.75)
  return (ndi.gaussian_filter(img, sigma=brush, mode='nearest') > 200).astype(img.dtype)

# Principal axis intersection
def raw_moment(data, i_order, j_order):
  """ Calculating raw moments for covariance matrix"""
  nrows, ncols = data.shape
  y_indices, x_indicies = np.mgrid[:nrows, :ncols]
  return (data * x_indicies**i_order * y_indices**j_order).sum()

def moments_cov(data):
  """ Returns covariance matrix """ 
  data_sum = data.sum()
  m10 = raw_moment(data, 1, 0)
  m01 = raw_moment(data, 0, 1)
  x_centroid = m10 / data_sum
  y_centroid = m01 / data_sum
  u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
  u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
  u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
  cov = np.array([[u20, u11], [u11, u02]])
  return cov

def principal_axis(img):
  """
    Returns a major axis and a minor axis coords

    Parameters
    ----------
    img : ndarray
        any image array.

    Returns
    -------
    major : dictionary
        Use p1 and p2 as key values to access coords.
    minor : dictionary
        Use p1 and p2 as key values to access coords.

    """
  cov = moments_cov(img)
  evals, evecs = np.linalg.eig(cov)
  sort_indices = np.argsort(evals)[::-1]
  x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
  x_v2, y_v2 = evecs[:, sort_indices[1]]

  scale = 150
  x1_red = x_v1*-scale*2; x2_red = x_v1*scale*2
  y1_red = y_v1*-scale*2; y2_red = y_v1*scale*2

  x1_blue = x_v2*-scale; x2_blue = x_v2*scale
  y1_blue = y_v2*-scale; y2_blue = y_v2*scale

  major = {'p1':[x1_red, y1_red], 'p2':[x2_red, y2_red]}
  minor = {'p1':[x1_blue, y1_blue], 'p2':[x2_blue, y2_blue]}
  
  return major, minor

def connect(ends):
    """argument ends is array [[x1,y1], [x2,y2]] """
    d0, d1 = np.abs(np.diff(ends, axis=0))[0]
    if d0 > d1: 
        return np.c_[np.linspace(ends[0][0], ends[1][0], d0+1, dtype=np.int32),
                     np.round(np.linspace(ends[0][1], ends[1][1], d0+1))
                     .astype(np.int32)]
    else:
        return np.c_[np.round(np.linspace(ends[0][0], ends[1][0], d1+1))
                     .astype(np.int32),
                     np.linspace(ends[0][1], ends[1][1], d1+1, dtype=np.int32)]

def intersectp(paxis,contour):
  """ Takes axis endpoint coords and contour coords 
      Returns : coords of intersecting points
  """
  axis_ends = np.array([paxis['p1'], paxis['p2']], dtype=np.int32)
  #print(line_ends)
  axis_line = connect(axis_ends)
  line1 = LineString(axis_line)
  line2 = LineString(contour)
  inter_sect = line1.intersection(line2)
  itr = np.array(inter_sect, dtype=np.int32) # here type of inter_sect was multipoint which was not iterable therefore i had to change it to numpy array
  return itr

def shift_to_start_point(contour, intersection_coords):
  distances = dict()
  for idx in range(len(contour)):
    for j in range(len(intersection_coords)):
      if(contour[idx][0] == intersection_coords[j][0]):
        if(contour[idx][1] == intersection_coords[j][1]):
          dist = sqrt(pow(contour[idx][0], 2) + pow(contour[idx][1], 2))
          #print("The index of intersection point {} in shape contour is : %d and distance from centroid is = %f".format(itr[j]) %(idx, dist))
          distances[idx] = dist
          roll = -(max(distances, key= distances.get))
          rolled = np.roll(contour, roll, axis=0)
  #print("Value to roll shape array by :", -(max(distances, key= distances.get)))
  return rolled

# CCD itself is translation invariant
def get_ccd(contour, xbar, ybar):
  """ returns ccd signature """
  di = []
  for i in range(len(contour)):    
      d = pow((contour[i][0] - xbar), 2) + pow((contour[i][1] - ybar), 2)
      di.append(sqrt(d))
  return di

def norm_ccd(signature):
  """ Normalises ccd for scale invariation """
  dis = []
  disum = sum(signature)
  for i in range(len(signature)):
    dis.append(signature[i] / disum)
  return dis


def feat_ext(image):
    smooth = extract_shape(image) # get shape of leaf
    # Step1: get contour of shape
    shape = get_contour(smooth)
    # Step2: find starting point based on principal axis intersection with contour
    shape = shift_to_origin(shape)
    mjr, mnr = principal_axis(smooth)
    itr_coords = intersectp(mjr, shape) 
    shape_rolled = shift_to_start_point(shape, itr_coords)
    # Step3: get the ccd signature
    cx, cy = get_center(shape_rolled)
    shape_sign = get_ccd(shape_rolled, cx, cy)
    norm_shape_sign = norm_ccd(shape_sign)
    