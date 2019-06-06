# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:38:12 2019

@author: kbergste

Module to hold all functions which do basic array operations that are useful
for the mms_feature_search script. Includes boxcar averaging, etc.
"""

import numpy as np

def boxcar_avg(array,width):
    '''
    Does boxcar averaging for an array over the number of data points
    Inputs:
        array- a numpy array of floats
        width- an integer, the number of data points averaging over
    Outputs:
        boxcar_avg- a boxcar avged array of the same dimension of the 
        input array
    '''
    weights=np.full(width,1.0)/width
    boxcar_avg=np.convolve(array,weights,mode='same')
    return boxcar_avg

def interval_mask(series,min_val,max_val):
    '''
    Takes numpy array data and returns a mask defined by given
    minimum and maximum values
    Inputs:
        series- the array to be masked (must have the < and > operations work)
        min_val- the lower bound value of the mask
        max_val- the upper bound value of the mask
    Outputs:
        mask- a boolean numpy array returning 1 where the series element is 
            between the given max/min values, and 0 where it is not
    ''' 
    mask=np.logical_and(series > min_val,series < max_val)
    return mask    

def find_avg_signs(array):
    '''
    Given an array of floats, determines the average sign of the data
    as positive or negative
    Inputs:
        array- the array of floats to find the average sign of
    Outputs:
        sign- 1 if average is positive, -1 if average is negative, 0 if 0
        quality- an indication of how good the calculation was
            1 is good, 0 is godawful
    ''' 
    total=np.nansum(array) #treat NaNs as zero
    abs_total=np.nansum(np.abs(array)) #treat NaNs as zero
    sign=np.sign(total)
    quality=abs(total/abs_total)
    return sign,quality

def fluct_abt_avg(array):
    '''
    Returns the array minus its average
    To show how it fluctuates
    Inputs:
        array- the numpy array of floats to find fluctuations of
    Outputs:
        fluct- the numpy array of fluctuations (same dimension as array)
    '''
    fluct=array - np.average(array)
    return fluct