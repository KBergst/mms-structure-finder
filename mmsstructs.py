# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:17:09 2019
@author: kbergste
module for all specialized sectioning functions for mms_feature_search.py
Does sectioning, structure categorization, etc. The meat of the analysis.
"""

import numpy as np
import scipy.signal as signal #for extrema finding

def section_maker(structure_extents,padding,max_index,min_index=0):
    '''
    Takes a tuple of indices of instances and makes a suitably-sized index
    window around each instance using the structure size, for plotting.
    Zoom in on the crossing itself
    Possibility of missing larger structure
    Inputs:
        structure_extents- list of 2-element lists which have the form
            [first structure index, last structure index]
        padding- the number of data points to add to each side of the window
        max_index- the maximum index in the data array. no data beyond it!
        min_index- the minimum index in the data array. defaults to zero
            all data in the data array have indices between min_index 
            and max_index
    Outputs:
        window_list- list of two-element lists, where each two-element list 
            corresponds to an index crossing, and has the endpoint indices
            for the plotting window around that crossing
    '''
    window_list=[]

    for structure in structure_extents:
        index_min=max(structure[0]-padding,min_index)
        index_max=min(structure[1]+padding,max_index)
        window_list.append([index_min,index_max])
    return window_list

def larger_section_maker(window,scale_factor,max_index,min_index=0):
    '''
    For select events, creates a window which is larger than the given window 
    by the window scale factor
    Inputs:
        window- the window to enlarge, created by section_maker, two-element
            list of the minimum and maximum indices of the window
        scale_factor- the amount the window is to be scaled by
        max_index- the maximum index of the data array (no data past it)
        min_index- the minimum index of the data array (default zero)
    Outputs:
        larger_window- two-element list of the minimum and maximum indices of
            the larger window
    '''
    half_width=int((window[1]-window[0])/2)
    min_window=max(window[0]-scale_factor*half_width,min_index)
    max_window=min(window[1]+scale_factor*half_width,max_index)
    larger_window=[min_window,max_window]
    return larger_window

def find_crossings(array,timeseries,gap_time):
    '''
    Finds indices of zero crossings for an array
    
    Also need to screen out crossings that happen in data gaps
    Could modify to find crossings for any value- not useful at the moment
    Inputs:
        array- the numpy float array to find crossings in
        timeseries- the timeseries associated with 'array' (for recording
                                                            the crossing times)
        gap_time- the minimum amount of time to be considered a data gap
    Outputs: 
        cleaned_indices- returns a list of the indices where crossings occur
            the convention is the index returned is the one to the direct
            left of the crossing
    '''
    arr_sign=np.sign(array)
    arr_compare=abs(arr_sign[:-1]-arr_sign[1:]) #compares each element to next one
    arr_crossing=(arr_compare == 2) #true if there is a crossing
    indices_rough=np.nonzero(arr_crossing) #indices with crossings to their
                                           #immediate right
    cleaned_indices=[]
    for n,index in enumerate(indices_rough[0]):
        if (timeseries[index+1]-timeseries[index] < gap_time) :
            cleaned_indices.append(index)
    return cleaned_indices

def find_crossing_signs(array,indices):
    '''
    Returns 1 if the crossing is from negative to positive 
    Returns -1 if the crossing is from positive to negative
    '''
    shift=np.array(indices)
    shift=shift+1
    shifted_indices=list(shift)
    return np.sign(array[shifted_indices]-array[indices])

def find_maxes_mins(array,indices,directions,width,min_height):
    '''
    Given list of zero crossings, finds maxes or mins between crossings
    Clean out crossings that don't attain at least 
    the minimum allowed crossing height
    Also make maxes/mins list for defining the size of the windows/structures
    Inputs:
        array- the data array in question
        indices- the list of indices where zero crossings occur
        directions- the direction of each crossing (neg to pos -> 1, 
                                                    pos to neg -> -1)
        width- the width (in data points) required to be a local extrema
        min_height- the minimum allowed crossing height (absolute value)
    Outputs:
        cleaned_indices-updated list of indices with insufficient crossings 
            screened
        directions[mask]-updated list-like object of directions with
            insufficent crossings screened
        max_indices- tuple of indices where local maxima occur
        min_indices- tuple of indices where local minima occur
    '''
    cleaned_indices_arr=np.array(indices) #to np array for processing purposes
    #find indices of mins and maxes for later window/structure processing
    max_indices_tmp=signal.argrelmax(array,order=width)
    min_indices_tmp=signal.argrelmin(array,order=width)
    #filter out mins/maxes which are not above the min_height
    max_indices=tuple(filter(lambda x: array[x] > min_height,
                             max_indices_tmp[0]))
    min_indices=tuple(filter(lambda x: -1*array[x] > min_height,
                             min_indices_tmp[0]))
    
    #filter out bad zero crossings using extrema heights
    prev_exts=np.array([])
    next_exts=np.array([])
    for n,direc in enumerate(directions):
        if direc > 0: #crossing from neg to pos
            if (n < len(indices)-1) and n>0 : #index on each side
                next_ext=np.amax(array[indices[n]:indices[n+1]])
                prev_ext=np.amin(array[indices[n-1]:indices[n]])
            elif (n < len(indices)-1): #first index
                next_ext=np.amax(array[indices[n]:indices[n+1]])
                prev_ext=np.amin(array[indices[n]-50:indices[n]])
            elif (n == len(indices)-1) and n>0: #last index
                next_ext=np.amax(array[indices[n]:indices[n]+50]) 
                prev_ext=np.amin(array[indices[n-1]:indices[n]])
            else: #both first and last index
                next_ext=np.amax(array[indices[n]:indices[n]+50]) 
                prev_ext=np.amin(array[indices[n]-50:indices[n]])
        else: #crossing from pos to neg
            if (n < len(indices)-1) and n>0 : #index on each side
                next_ext=np.amin(array[indices[n]:indices[n+1]])
                prev_ext=np.amax(array[indices[n-1]:indices[n]])
            elif (n < len(indices)-1): #first index
                next_ext=np.amin(array[indices[n]:indices[n+1]])
                prev_ext=np.amax(array[indices[n]-50:indices[n]])
            elif (n== len(indices)-1) and n>0: #last index
                next_ext=np.amin(array[indices[n]:indices[n]+50]) 
                prev_ext=np.amax(array[indices[n-1]:indices[n]])
            else: #both first and last index
                next_ext=np.amin(array[indices[n]:indices[n]+50])
                prev_ext=np.amax(array[indices[n]-50:indices[n]])
        prev_exts=np.append(prev_exts,[prev_ext]) #add to list
        next_exts=np.append(next_exts,[next_ext]) #add to list

    #Implement (complicated, inefficient) selection mechanism
    #will probably need to be explained in words (or flowchart)
    mask=np.array([],dtype=bool)
    for n,(direc,prev_ext,next_ext) in enumerate(zip(directions,
                                                   prev_exts,next_exts)):
        if (abs(prev_ext)<min_height):
            mask=np.append(mask,[False])
        elif (abs(prev_ext)>=min_height and 
            abs(next_ext)>=min_height):
            mask=np.append(mask,[True])
        else:
            i=n+1
            while i<len(directions):
                if (abs(next_exts[i])>=min_height and 
                    directions[i]==direc): #trend is continuing up or down across 0
                    mask=np.append(mask,[True])
                    break
                elif (abs(next_exts[i])>=min_height): #up then down or vice versa
                    mask=np.append(mask,[False])
                    break
                i+=1
                
    cleaned_indices=list(cleaned_indices_arr[mask]) #put back to list form
    return cleaned_indices,directions[mask],max_indices,min_indices

def structure_extent(indices,times,directions,maxes,mins,max_index,
                     min_index=0):
    '''
    Like section_maker, but intended to determine the actual spatial extent
    Of the structure (using the direction of the crossing and nearest max/min)
    Inputs:
        indices- list of indices where crossings occur in the data array
        times- numpy array of the timeseries of the array which has zero crossings
        directions- list of directions of the crossings
        maxes- tuple of indices where local maximums occur in the data array
        mins- tuple of indices where local minimums occur in the data array
        max_index- the maximum index of the data array (no data past it)
        min_index- the minimum index of the data array (default zero)
    Outputs:
        size_list- list of two-element lists which contain the minimum and 
            maximum index for each structure
        times_list- list of two-element lists which contain the minimum and 
            maximum time for each structure
    '''
    size_list=[]
    times_list=[]
    mins_arr=np.asarray(mins) #tuple to numpy array for using searches
    maxes_arr=np.asarray(maxes)

    for n,current_index in enumerate(indices):
        if directions[n]==1: #crossing from negative to positive
            left_idx=np.searchsorted(mins_arr,current_index,side='left')-1
            right_idx=np.searchsorted(maxes_arr,current_index,side='left')
            index_min=mins_arr[left_idx]
            index_max=maxes_arr[right_idx]
        else: #crossing from positive to negative
            left_idx=np.searchsorted(maxes_arr,current_index,side='left')-1
            right_idx=np.searchsorted(mins_arr,current_index,side='left')  
            index_min=maxes_arr[left_idx]
            index_max=mins_arr[right_idx]
        size_list.append([index_min,index_max])
        times_list.append([times[index_min],times[index_max]])
    return size_list,times_list

def structure_classification(crossing_direction,vex_direction,vex_quality,
                             jy_direction,jy_quality,q_min):
    '''
    Determines the tentative kind of structure using:
    -The direction of the Bz crossing
    -The sign of vex
    -The sign of jy
    If the determination of the signs of vex or jy are not of a certain quality
    The classification reteurns 'uncertain'
    Inputs:
        crossing_direction- the direction of the crossing
        vex_direction- the direction of the GSM x-component of the electron
            velocity at the crossing
        vex_quality- the quality of the determination of vex_direction
        jy_direction- the direction of the GSM y-component of the electron
            velocity at the crossing
        jy_quality- the quality of the determination of jy_direction
        q_min- minimum acceptable quality factor
    Outputs:
        string output- the string classification of the object
        integer output- the integer flag of the object type
    '''
    if((jy_quality < q_min) or (vex_quality < q_min)):
        return "Uncertain- low quality",3
    elif ((jy_direction>0) and (vex_direction>0) and (crossing_direction>0)):
        return "Plasmoid moving earthward",0 
    elif ((jy_direction>0) and (vex_direction<0) and (crossing_direction<0)):
        return "Plasmoid moving tailward",0 
    elif ((jy_direction>0) and (vex_direction>0) and (crossing_direction<0)):
        return "Pull Current sheet moving earthward",1 
    elif ((jy_direction>0) and (vex_direction<0) and (crossing_direction>0)):
        return "Pull Current sheet moving tailward",1
    elif ((jy_direction<0) and (vex_direction>0) and (crossing_direction<0)):
        return "Push current sheet moving earthward",2
    elif ((jy_direction<0) and (vex_direction<0) and (crossing_direction>0)):
        return "Push current sheet moving tailward",2
    else:
        return "Uncertain- didn't match any given case",4
    
def structure_sizer(endtimes,velocs):
    '''
    Determines the approximate x- size of the structure using start and stop 
    times and the velocity series of the structure (use curlometer ve for now)
    Inputs:
        endtimes list of start and stop times (datetime objects)
        velocs- numpy array of velocities in km/s
    Outputs:    
        returns size in km
    '''    
    time_interval=(endtimes[1]-endtimes[0]).total_seconds()
    speed_avg=abs(np.average(velocs))
    
    return speed_avg*time_interval
