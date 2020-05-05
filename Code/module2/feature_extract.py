# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:45:17 2020

@author: Ganesh
"""
from module2 import shape as shp
from module2 import margin as mrg
from module1 import postprocess
def feat_ext(image):
    smooth = shp.extract_shape(image) # get shape of leaf
    # Step1: get contour of shape
    shape = shp.shift_to_origin(postprocess.get_contour(smooth))
    # Step2: find starting point based on principal axis intersection with contour
    mjr, mnr = shp.principal_axis(smooth)
    itr_coords = shp.intersectp(mjr, shape) 
    shape_rolled = shp.shift_to_start_point(shape, itr_coords)
    # Step3: get the ccd signature
    cx, cy = shp.get_center(shape_rolled)
    shape_sign = shp.get_ccd(shape_rolled, cx, cy)
    norm_shape_sign = shp.norm_ccd(shape_sign)
    interp_shape_sign = shp.interp_ccd(norm_shape_sign)

    return norm_shape_sign, interp_shape_sign
    #med_blur = mrg.extract_margin(image)