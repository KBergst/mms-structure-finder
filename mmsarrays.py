# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 09:38:12 2019

@author: kbergste

Module to hold all functions which do basic array operations that are useful
for the mms_feature_search script. Includes boxcar averaging, etc.
"""

import numpy as np

def angle_between(array1,array2):
    '''
    Determines the angle between two vectors in the typical 3d way
    Inputs:
        array1- the first array to be used
        array2- the second array to be used
    Outputs:
        angle- the angle in radians between the two arrays 
            domain is -pi/2 to pi/2
    '''
    angle=np.arccos(np.dot(array1,array2)/ \
                    (np.linalg.norm(array1)*np.linalg.norm(array2)))
    
    if angle > np.pi/2 : #change arccos domain from [0,pi] to [-pi/2,pi/2]
        angle=np.pi-angle
    
    return angle

def vector_projection(array,norm):
    '''
    Returns the projection of array onto the plane which is normal to the norm
        vector
    Inputs:
        array-the array to project onto the plane
        norm- the normal to the plane
    Outputs:
        plane_proj- the projection onto the plane
    '''    
    norm_proj=np.dot(array,norm)/np.dot(norm,norm)*norm
    plane_proj=array-norm_proj
    
    return plane_proj

def coord_transformation(vectors,basis_matrix):
    '''
    Transforms a vector into the coordinates specified by the given
        basis matrix
    Inputs:
        vector-the vectors to be transformed, in an array of shape (length,3)
        basis_matrix- the basis matrix of the new coordinates
    Outputs:
        new_vector- vector in the new coordinates
        
    TODO: make this more efficient if possible, there has to be a better way
    '''
    basis_matrix_inv=np.linalg.inv(basis_matrix)
    new_vectors=np.transpose(np.array([[],[],[]]))
    for i in range(len(vectors[:,0])):
        vector=vectors[i,:]
        new_vector=np.reshape(np.matmul(basis_matrix_inv,vector),(1,3))
        new_vectors=np.append(new_vectors,new_vector,axis=0)
        
    return new_vectors

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

def hamming_smooth(array,width):
    '''
    Does hamming window smoothing of an array of data
    Inputs:
        array- input array to do smoothing on, should not be multidiml
            (do each component of vector quantities separately)
        width- the width of the hamming window to use
    Outputs:
        smoothed_arr-array smoothed using the hamming window
    '''
    
    #reduce boundary effects at the beginning/end of the array
    reflected_arr=np.r_[array[width-1:0:-1],array,array[-2:-width-1:-1]]
    #do smoothing
    hamming=np.hamming(width)
    y=np.convolve(hamming/hamming.sum(),reflected_arr,mode='valid') #wrong dimension
    smoothed_arr=y[(width//2-1):-(width//2)]      
    
    return smoothed_arr

def smoothing(array):
    '''
    for determining how large of a window to do hamming smoothing over
    sets a baseline ratio of 0.1 window/total, rounded to an even integer
    returns the smoothed array
    Inputs:
        array- input array for smoothing, can be multidiml of shape
            (datlength,n) for n the number of different components
    Outputs:
        
    '''
    width=(len(array[:,0])//10)*2
    
    if width < 3:
        #too small for meaningful smoothing
        print("calculated smoothing width too small")
        return array
    
    smoothed_array=np.empty_like(array)
    for n in range(len(array[0,:])):
        smoothed_array[:,n]=hamming_smooth(array[:,n],width)
        
    return smoothed_array
    
    
    
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

def nested_mva(array):
    '''
    Does a nested minimum-variance analysis procedure on an array of 
        3-dimensional data given.
    Inputs:
        array- array of data of shape (datlength,3)
    Outputs:
        eigenvals_final- array of the three final eigenvalues from the analysis
            ordered from largest to smallest
                0- maximum variance
                1- intermediate variance
                2- minimum variance (normal direction for a current sheet )
        eigenvecs_final- list of the three final eigenvectors (in array form)
        angle_errs- array containing vectors of uncertainties of the angles 
            between the direction pairs: 01, 02, and 12.
        angle_ref_deviation_0-angular displacement of the nest normal from the
            reference normal, in the direction of reference direction 0
        angle_ref_deviation_1-angular displacement of the nest normal from the
            reference normal, in the direction of reference direction 1
    '''    
    halfway_point=len(array[:,0])//2
    if (len(array[:,0]) % 2 == 0):
        num_nests=halfway_point-1
    else:
        num_nests=halfway_point

#    avging_points=int(len(array[:,0])*smoothing)
#    array_smooth=np.transpose(np.array([boxcar_avg(column,avging_points) for \
#                                        column in array.T]))   
    junk1,tmp,junk2=basic_mva(array) #reference norm
    ref_dir_norm=tmp[:,2] # reference minimum variance direction
    ref_dir_0=tmp[:,0] # reference maximum variance direction
    ref_dir_1=tmp[:,1] # reference intermediate variance direction 
    eigenvals=[]
    eigenvecs=[]
    angle_errs=np.transpose(np.array([[],[],[]]))
    points_num=np.array([]) #number of points in each nest
    angle_02=np.array([]) #for deviation of nest norm from ref norm, in the 0 direction
    angle_12=np.array([]) #for deviation of nest norm from ref norm, in the 1 direction
    
    #calculate each nest's minimum variance
    for i in range(num_nests):
        points_num=np.append(points_num,[2*i+3])
        subarray=array[halfway_point-1-i:halfway_point+2+i,:]
        tmp1,tmp2,tmp3=basic_mva(subarray)
        eigenvals.append(tmp1)
        eigenvecs.append(tmp2)
        nest_angle_errs=np.reshape(tmp3,(1,3))
        angle_errs=np.append(angle_errs,nest_angle_errs,axis=0)
        plane_02_norm_proj=vector_projection(tmp2[:,2],ref_dir_1) #02 plane
        plane_12_norm_proj=vector_projection(tmp2[:,2],ref_dir_0) #12 plane
        angle_02=np.append(angle_02,
                           [angle_between(plane_02_norm_proj,ref_dir_norm)])
        angle_12=np.append(angle_12,
                           [angle_between(plane_12_norm_proj,ref_dir_norm)])
               
    '''
    for now, just take the eigenvalues/vectors from the largest nest and use 
    the angles references and angle_errs to determine how good the estimate is    
    '''
    eigenvals_final=eigenvals[-1]
    eigenvecs_final=eigenvecs[-1]
    
    return eigenvals_final,eigenvecs_final,angle_errs,points_num,angle_02, \
            angle_12
            
            
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
    
    delt_phi=np.array([delt_phi01,delt_phi02,delt_phi12])
    
    return delt_phi

    
    