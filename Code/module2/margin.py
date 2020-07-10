import numpy as np
import cv2 as cv

def extract_margin(img):
  size = int(np.count_nonzero(img)/1000)
  if (size%2 == 1):
      new_size = size
  else:
      new_size = size+1
  return(cv.medianBlur(img, new_size))