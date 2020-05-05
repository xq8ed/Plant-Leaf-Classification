# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:45:17 2020

@author: Ganesh
"""
import shape
import margin
def feat_ext(image):
    smooth = shape.extract_shape(image) # get shape of leaf
    # Step1: get contour of shape
    shape = shape.shift_to_origin(get_contour(smooth))
    # Step2: find starting point based on principal axis intersection with contour
    mjr, mnr = shape.principal_axis(smooth)
    itr_coords = shape.intersectp(mjr, shape) 
    shape_rolled = shape.shift_to_start_point(shape, itr_coords)
    # Step3: get the ccd signature
    cx, cy = shape.get_center(shape_rolled)
    shape_sign = shape.get_ccd(shape_rolled, cx, cy)
    norm_shape_sign = shape.norm_ccd(shape_sign)

    med_blur = margin.extract_margin(image)