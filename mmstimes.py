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
import pytz #for timezone info


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
