# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:02:42 2019

@author: kbergste

module containing functions which will deal with reading data from files
"""
import numpy as np
import cdflib #NEEDS python 3.7 to run

import mmstimes as mt

def filenames_get(name_list_file):
    '''
    Pulls list of filenames I'm using from the file where they are stored.
    Allows some flexibility
    Inputs:
        name_list_file- string which constains the full path to the file which
            contains a list of the full filename paths needed
    Outputs:
        name_list- list of strings which contain the full path to 
            each file
    '''
    name_list=[]
    with open(name_list_file,"r") as name_file_obj: #read-only access
         for line in name_file_obj:
             line_clean =line.rstrip('\n') #removes newline chars from lines
             name_list.append(line_clean)
    return name_list 
    
    
def get_cdf_var(filename,varnames):
    """
    pulls particular variables from a CDF
    note: if variable has more than one set of data (E.g. b-field with x,y,z
    components) it will be necessary to format the data by reshaping the array
    from a 1D to a 2D array
    (may find workaround/better way later)
    Inputs:
        filename- string of the complete path to the specified CDF file
        varnames- list of strings which contain the CDF variables that are
            to be extracted
    Outputs:
        data- list of numpy arrays containing the desired variables' data
    """
    cdf_file=cdflib.CDF(filename,varnames)
    data=[]
    for varname in varnames:
        var_data=np.array(cdf_file.varget(varname))
        
        data.append(var_data)
    return data

def import_jdata(filename):
    '''
    Imports current density data outputted from the mms_curlometer script.
    Format is time (string format),jx,jy,jz (A)
    Inputs:
        filename- string of the complete path to the specified file
    Outputs:
        time- numpy array of datetime objects
        j_data- numpy array of j data (jx,jy,jz) in microAmps
    TODO: this is slow. Consider putting the jdata into CDF instead of text
        form, or format the datetime outputs so that it isn't necessary to use
        'parse', which I imagine is inefficient
    '''
    amps_2_uamps=1e6
    time_str=np.loadtxt(filename,delimiter=',',usecols=[0],dtype="str")
    j_data=np.loadtxt(filename,delimiter=',',usecols=range(1,4))*amps_2_uamps
    time_clean=[]
    for t in time_str:
        time_dt=mt.str2datetime(t)
        time_clean.append(time_dt)
    time=np.array(time_clean)
    return time,j_data

