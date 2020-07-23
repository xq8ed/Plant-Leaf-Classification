"""post-processing on cropped binary-image"""
import cv2 as cv
import numpy as np
import scipy.ndimage as ndi

def get_contour(img):
    """returns the coords of the longest contour"""
    cnt = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE) [-2]
    max_cnt = max(cnt, key = len).reshape(-1,2)
    return max_cnt

#--- Removing noise
def rem_noise(image):
    """ Returns noise free contour """
    struct = [[ 0., 0., 1., 1., 0., 0.],
              [ 0., 1., 1., 1., 1., 0.],  
              [ 1., 1., 1., 1., 1., 1.], 
              [ 1., 1., 1., 1., 1., 1.], 
              [ 1., 1., 1., 1., 1., 1.], 
              [ 0., 1., 1., 1., 1., 0.],
              [ 0., 0., 1., 1., 0., 0.]]
    
    erosion = get_contour(ndi.morphology.binary_erosion(image > 254, structure=struct).astype(image.dtype))
    closing = get_contour(ndi.morphology.binary_closing(image > 254, structure=struct).astype(image.dtype))
    opening = get_contour(ndi.morphology.binary_opening(image > 254, structure=struct).astype(image.dtype))
    dilation = get_contour(ndi.morphology.binary_dilation(image > 254, structure=struct).astype(image.dtype))
    
    return opening

def postp(img, flag = False):
    mask = np.zeros(img.shape[:2], np.uint8)
    noise_free_cnt = rem_noise(img)
    cv.drawContours(mask, [noise_free_cnt], 0, 255, -1)     #--- Applying noise free contour to mask
    return mask