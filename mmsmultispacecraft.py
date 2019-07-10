# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:28:44 2019

Home of functions to perform various multispacecraft techniques on MMS data
@author: kbergste
"""

import numpy as np
import datetime as dt

def bartlett_interp(array,times,new_times,two_dt_s=1/128,two_DT_s=1/128,q=1):
    '''
    Does the numerical Bartlett window described in Analysis Methods for 
    Multi-Spacecraft Data, Chapter 2 for interpolation of data onto new time tags.
    Generally will use q=1 to have 4 data points (in general) for
    DO NOT USE if times and newtimes are not basically the same data rate!!!!
    
    Inputs:
        array- the data array (e.g. Bz)
            can be multiple data at once in form (datlength,3) e.g. for Bx,By,Bz
        times- the time tags for the data array
        newtimes- the times desired to interpolate the array onto
        two_dt_s- the time cadence of the new timeseries, default 1/128s, the MMS cadence
        two_DT_s- the time cadence of the time tags (times), default 1/128s
        q- the q-factor of the Bartlett window (default 1)
    Outputs:
        new_array- the new data array in form (newdatlength,3)
    '''
    #convert time cadences to timedeltas
    two_dt=dt.timedelta(seconds=two_dt_s)
    
    new_array=np.empty((0,len(array[0,:])))
    for new_time in new_times: 
        l=np.searchsorted(times,new_time-2*q*two_dt,side="right")
        m=np.searchsorted(times,new_time-two_dt/2.,side="left")
        n=np.searchsorted(times,new_time+2*q*two_dt,side="left")-1
        
        x=(new_time-times[m]).total_seconds()/two_DT_s
        
        #determine normalization coefficient
        p=int(2*q+1/2)
        Se=2*p-p*p/2/q
        if (m-l == n-m): #odd number of points
            if ((2*q %1)  < 1/2): #as per eqn 2.20 in Analysis Methods for...
                S=Se+((2*q %1)-abs(x))/2/q
            else:
                S=Se+(1-(2*q %1)-abs(x))/2/q
        else:
            S=Se
        #determine the weights for each datum in the array
        weights=np.zeros(len(times),dtype="float")
        for k in range(1,m-l+1):  #as per eqn 2.15 in Analysis Methods for...
            weights[m-k]=1/S*(1-(k+x)/2/q)
        weights[m]=1/S*(1-abs(x)/2/q)
        for k in range(1,n-m+1):
            weights[m+k]=1/S*(1-(k-x)/2/q)
        
        #apply weights to each array of data
        tmp=weights@array
        new_values=tmp.reshape(1,len(tmp))
        new_array=np.concatenate((new_array,new_values))
        
    return new_array
        
        
        