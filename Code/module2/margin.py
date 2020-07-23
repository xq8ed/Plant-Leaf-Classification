""" Generating feature vector for margin """
import numpy as np
import cv2 as cv
from math import sqrt, atan2
from statistics import mean
from shapely.geometry import LineString
from module2.shape import shift_to_origin, interp_ccd
from module1.postprocess import get_contour

def blur_size(img):
  size = int(np.count_nonzero(img)/1000)
  if (size % 2 == 1):
    new_size = size
  else:
    new_size = size + 1
  return new_size

def margin_dist(blur_cont, original_cont):

  L = len(blur_cont)
  dist = []
  scale = 100 # constant
  cont_line = LineString(original_cont) # constant

  for i in range(L):
    nxt = np.mod(i+10, L)
    A = blur_cont[i-10]
    B = blur_cont[nxt]
    C = blur_cont[i]
    k = ((B[1]-A[1]) * (C[0]-A[0]) - (B[0]-A[0]) * (C[1]-A[1])) / (pow((B[1]-A[1]), 2) + pow((B[0]-A[0]), 2)) # slope
    D = np.array([C[0] - k * (B[1]-A[1]), C[1] + k * (B[0]-A[0])]) # finding intersection coords on line AB from C

    if(np.array_equal(D,C)): # if point C is on line AB
      if(B[1]-A[1] == 0): # line AB is horizontal
        x1 = x2 = D[0]
        y1 = D[1] + 100; y2 = D[1] - 100
      elif(B[0] -A[0] == 0): # line AB is vertical
        x1 = D[0] - 100; x2 = D[0] + 100
        y1 = y2 = D[1]
      else:
        slope = (B[1]-A[1]) / (B[0]-A[0])
        slope = -1/slope
        x1 = D[0] - 100
        x2 = D[0] + 100
        y1 = slope * -100 + D[1]
        y2 = slope * 100 + D[1]

      DC = LineString(np.array([[x1, y1], [x2, y2]]))
    else:
      dx = C[0] - D[0]; dy = C[1] - D[1] # distance along x and y direction between point D and C
      lenDC = sqrt(pow(dx, 2) + pow(dy, 2)) # length vector from D to C
      Xvec = (dx) / lenDC
      Yvec = (dy) / lenDC
      x2 = C[0] + Xvec * scale
      y2 = C[1] + Yvec * scale
      x1 = D[0] - Xvec * scale
      y1 = D[1] - Yvec * scale
      DC = LineString(np.array([[x1, y1], [x2, y2]]))

    intersection = DC.intersection(cont_line)
    try: # when there are more than one intersection points
      itr1 = [np.array(pp.coords, dtype=np.float32) for pp in intersection] # converting LineString Object to numpy array
      itr2 = np.hstack(ii.reshape(1, -1) for ii in itr1)
      itr3 = itr2.reshape(int(len(itr2[0])/2), 2)
      itr4 = np.unique(itr3, axis=0)
      d = min(np.array([sqrt(pow((x[0]-C[0]), 2) + pow((x[1]-C[1]), 2)) for x in itr4]))
      #print("More than one intersection points", i)
    except TypeError:
      itr4 = np.array(intersection.coords[0])
      d = sqrt(pow((itr4[0]-C[0]), 2) + pow((itr4[1]-C[1]), 2))
      #print("Single point : ", i)
    dist.append(d)
  return dist

def partition(margin_sign):
  parts = []
  L = len(margin_sign)
  win_size = int(L / 100)
  overlap = win_size - int(win_size/2)
  parts = [margin_sign[i:i+win_size] for i in range(0, len(margin_sign), overlap)]
  return parts

def magnitude(margin_sign, med_blur_img, blade_img):
  mag_list = margin_sign
  mb_cont = get_contour(med_blur_img)
  mb_cont_x = mb_cont[::,0]
  mb_cont_y = mb_cont[::,1]
  blade_inv_ix = blade_img[mb_cont_y, mb_cont_x] # assigning values to boundary points of med_blur which fall inside leaf region
  for i in range(0,len(blade_inv_ix)):
    if (blade_inv_ix[i] == 255):
      mag_list[i] = margin_sign[i] * -1 # assigning - sign to points falling inside of shape(smooth1)
  return partition(mag_list)

def gradient(margin_sign):
  windows = partition(margin_sign)
  for j in range(0, len(windows)):
    windows[j] = [x - windows[j][i - 1] for i, x in enumerate(windows[j])][1:]
  return windows

def getAngle(a, b, c):
  """ Returns angle at b in radians with sign indiacting direction in which angle is measured """
  ang = atan2(c[1]-b[1], c[0]-b[0]) - atan2(a[1]-b[1], a[0]-b[0])
  return ang

def curvature(blur_img_cont):
  L = len(blur_img_cont)
  angles = []
  for i in range(L):
    currentp = blur_img_cont[i]
    nxt = np.mod(i+1, L)
    nextp = blur_img_cont[nxt]
    prevp = blur_img_cont[i-1]
    angles.append(getAngle(prevp, currentp, nextp))
  return partition(angles)

def avg(lst):
  plist = []
  nlist = []
  for i in range(0, len(lst)):
    plist.append([y for y in lst[i] if y>=0])
    nlist.append([y for y in lst[i] if y<=0])
  avg_p = []
  avg_n = []
  for a, b in zip(plist, nlist):
    if len(a)>0:
      avg_p.append(mean(a))
    else:
      avg_p.append(0)
    if len(b)>0:
      avg_n.append(mean(b))
    else:
      avg_n.append(0)
  return avg_p, avg_n

def norm_margin(sign):
  """
  Input: vectorized sign
  Output: normalized sign
  """
  lst = []
  for vec in sign:
    lst.append(mean(vec))
  return lst

def margin_vector(img, flag = False):
	size = blur_size(img)
	med_blur = cv.medianBlur(img, size)
	medblur_cont = shift_to_origin(get_contour(med_blur))
	outline_cont = shift_to_origin(get_contour(img))
	sign = margin_dist(medblur_cont, outline_cont)
	
	mag = magnitude(sign, med_blur, img)
	grad = gradient(sign)
	curve = curvature(medblur_cont)

	ap_mag, an_mag = avg(mag)
	ap_grad, an_grad = avg(grad)
	ap_curve, an_curve = avg(curve)
	vec_sign = list(zip(ap_mag, an_mag, ap_grad, an_grad, ap_curve, an_curve))
	norm_margin_sign = norm_margin(vec_sign)
	interp_margin_sign = interp_ccd(norm_margin_sign)

	return interp_margin_sign