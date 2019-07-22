# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:41:31 2019

@author: kbergste

For converting to and from the MMS time conventions (TT 2000, nanoseconds)
Currently can convert to/from datetime objects, but can add functions with
additional conversions later if needed/desired.
"""

import matplotlib as mpl
import datetime as dt
import numpy as np
import pytz #for timezone info
import spacepy.coordinates as spcoords
from spacepy.time import Ticktock


def TTtime2datetime(time_nanosecs):
    """
    converts MMS CDF times (nanoseconds, TT 2000) to 
    matplotlib datetime objects
    Inputs:
        time_nanosecs- time straight from MMS data
    Outputs:
        t_utc- time in UTC as a datetime object
    """    
    start_date=dt.datetime(2000,1,1,hour=11,minute=58,second=55,
                                 microsecond=816000)
        # see https://aa.usno.navy.mil/faq/docs/TT.php for explanation of
        # this conversion time
    start_num=mpl.dates.date2num(start_date) #num of start date of time data
    fudge_factor=5/60/60/24 #correction amount (correcting for leap seconds?)
    time_days=time_nanosecs/1e9/60/60/24 #convert nanoseconds to days
    time_num=time_days+start_num-fudge_factor #convert TT2000 to matplotlib num time
    times=[]
    if time_num.size>1:
        for t in time_num:
            t_dt=mpl.dates.num2date(t) #conversion to datetime object
            t_utc=t_dt.astimezone(pytz.utc) #officially set timezone to UTC
            times.append(t_utc)
        return times        
    else:
        t_dt=mpl.dates.num2date(time_num) #conversion to datetime object
        t_utc=t_dt.astimezone(pytz.utc) #officially set timezone to UTC
        return t_utc
    
def datetime2TTtime(time_dt):
    '''
    converts datetime objects to MMS CDF times (nanoseconds,TT 2000)
    Inputs:
        time_dt- time as a datetime object
    Outputs:
        time_nanosecs- time in nanoseconds since J2000 
    '''
    start_date=dt.datetime(2000,1,1,hour=11,minute=58,second=55,
                                 microsecond=816000)
        # see https://aa.usno.navy.mil/faq/docs/TT.php for explanation of
        # this conversion time
    start_num=mpl.dates.date2num(start_date)  #num of start date of time data
    fudge_factor=5/60/60/24 #correction amount (correcting for leap seconds?)
    
    time_num=mpl.dates.date2num(time_dt)
    time_days=time_num-start_num+fudge_factor
    time_nanosecs=time_days*24*60*60*1e9
    return time_nanosecs

def datetime2str(time_dt):
    '''
    converts datetime objects to standardized strings of format
        %Y-%m-%d %H:%M:%S.%f%z (see datetime info on what this is)
    Inputs:
        time_dt- time as a datetime object        
    Outputs:
        time_str= time as a string
    '''
    time_str=time_dt.strftime("%Y-%m-%d %H:%M:%S.%f%z")
    
    return time_str

def str2datetime(time_str):
    '''
    converts strings of format %Y-%m-%d %H:%M:%S.%f%z into datetime objects
    
    Inputs:
        time_str- time as a string of the above format
    Outputs:
        time_dt- time as a datetime object
    '''
    time_dt=dt.datetime.strptime(time_str,'%Y-%m-%d %H:%M:%S.%f%z')
    
    return time_dt  

def datetime_avg(times):
    '''
    finds the average datetime out of a bunch
    basically just converts to TTtime averages, then puts it back
    Inputs:
        times- the array of datetime objects to be averaged
    Outputs:
        avg_time- the average time represented as a datetime object
    '''
    TT_times=datetime2TTtime(times)
    avg_TT_time=np.average(TT_times)
    avg_time=TTtime2datetime(avg_TT_time)
    
    return avg_time

def coord_transform(vecs,sys1,sys2,times):
    '''
    Uses spacepy to do coordinate transformations (assuming cartesian)
    Inputs:
        vecs- the vector of arrays in the old coordinates of shape (datlength,3)
        sys1- string denoting the old coordinates e.g. 'GSE','GSM', etc.
        sys2- string denoting the new coordinates e.g. 'GSE','GSM', etc.
        times-array of timestamps for the vecs, as datetime objects in UTC time
    Outputs:
        new_vecs- the vector of arrays in the new coordinates of shape 
            (datlength,3)
    '''
    cvals=spcoords.Coords(vecs,sys1,'car')
    cvals.ticks=Ticktock(times,'UTC')
    new_cvals=cvals.convert(sys2,'car')
    new_vecs=new_cvals.data
    
    return new_vecs
    