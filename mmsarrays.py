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

def basic_mva(array):
    '''
    Does a basic minimum-variance analysis step on array of 3-dimensional data
        given.
    Inputs:
        array- array of data of shape (datlength,3)
    Outputs:
        eigenvals- array of the three eigenvalues from the analysis
            ordered from largest to smallest
        eigenvecs- list of the three eigenvectors (in array form)
        angle_errs- vector of uncertainties of the angles between the direction
            pairs: 01, 02, and 12.
    '''
    
    avgs=np.average(array,axis=0) #averages each component
    Mb=np.empty((3,3))
    for i in range(3): #probably too slow, could be improved
        for j in range(3):
            Mb[i,j]=np.average(array[:,i]*array[:,j])-avgs[i]*avgs[j]
            
    #compute eigenvectors and values of Mb, and sort them by decreasing magnitude
    tmp1,tmp2=np.linalg.eig(Mb) #column tmp2[:,i] is the eigenvector
    idx=tmp1.argsort()[::-1] #default sort is small to big, so need to reverse
    eigenvals=tmp1[idx]
    eigenvecs=tmp2[:,idx]
    
    #find the statistical errors
    angle_errs=stat_err_mva(eigenvals,array.shape[0])
    
    return eigenvals, eigenvecs, angle_errs
    
def stat_err_mva(eigenvals,M):
    '''
    Finds the statistical uncertainty associated with a particular mva 
    calculation
    used by the basic_mva function
    Inputs:
        eigenvals- array of the three eigenvalues from the analysis
            ordered from largest to smallest
        M- the number of data points the MVA is being done on
    Outputs:
        delt_phi- vector of uncertainties of the angles between the direction
            pairs: 01, 02, and 12.
    '''
    coef=np.sqrt(eigenvals[2]/(M-1))
    delt_phi01=coef*np.sqrt((eigenvals[0]+eigenvals[1]-eigenvals[2])/ \
                            (eigenvals[0]-eigenvals[1])**2)
    delt_phi02=coef*np.sqrt((eigenvals[0]+eigenvals[2]-eigenvals[2])/ \
                            (eigenvals[0]-eigenvals[2])**2)    
    delt_phi12=coef*np.sqrt((eigenvals[1]+eigenvals[2]-eigenvals[2])/ \
                            (eigenvals[1]-eigenvals[2])**2)
    
    delt_phi=[delt_phi01,delt_phi02,delt_phi12]
    
    return delt_phi
    
    