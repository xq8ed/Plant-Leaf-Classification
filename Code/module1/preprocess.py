""" module for preprocessing on raw image """

import cv2 as cv
import numpy as np

def prep(path, flag = False):
    """
    Input : path to image file
    Returns : binary image
    """

    img = cv.imread(path)
    
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    #--- Converting to grayscale ---
    gray = cv.cvtColor(hsv_img, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #--- Converting to bianry ---
    blur = cv.GaussianBlur(gray, (5,5), 0)
    #canny = cv.Canny(blur, 50, 100)
    retval, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
    
    #--- finding longest contour ---
    contours = cv.findContours(binary.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) [-2] # [-2] is offset used, otherwise drawContours doesn't recognise contour
    contour = max(contours, key = len) # Finding Contour with longest lenth
    
    # --- masking ---
    mask_bin = np.zeros(img.shape[:2], np.uint8) # Creating blank background with 0 intensity(dark) with resolution equal to source image
    cv.drawContours(mask_bin, [contour], 0, 255, -1) # the [] around contour and 3rd argument 0 means only the particular contour is drawn
    masked_gray = cv.bitwise_and(gray2, gray2, mask= mask_bin) # Removing background from grayscale image for texture extraction
    
    #--- cropping binary image ---
    #print("Cropped mask after applying to binary image :")
    x, y, w, h = cv.boundingRect(contour)
    crop_bin = mask_bin[y-20:y+h+20, x-20:x+w+20]
    
    #--- cropping gray image ---
    #print("Cropped gray image after applying mask :")
    crop_gray = masked_gray[y-20:y+h+20, x-20:x+w+20]
    
    return crop_bin