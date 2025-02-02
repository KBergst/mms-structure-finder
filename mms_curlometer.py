#full_curlometer.py code adapted for use with MMS CDF files
#import sys #for stopping while debugging
import numpy as np  # for matrix calculations
import math # for pi and sqrt
#import glob # for sensible listdir()
import cdflib #for importing cdf files
from copy import deepcopy # for obtaining variables in CEF files
#import matplotlib.pyplot as plt # for plotting
import datetime as dt # for dates
from matplotlib import dates # for formatting axes
import pytz #for my own time stuff
import scipy.interpolate as interp #for interpolating to MMS1 timeseries
import os #for generalization to all systems
import time as tm #for checking runtime

#user created modules:
import mmstimes as mt

start=tm.time() 

# User-defined variables:

# Base path:
path = os.getcwd()
# File location key name:
loc_file = r"key_for_curlometer.txt"
# Output file location key name:
out_loc_file = r"curlometer_files.txt"

#directories for outputs, input data:
outpath="Curlometer_data"
datpath="MMS"
# output file start for output current density (minus number):
outfile = os.path.join(path,outpath,"output")
# output file location key (full path)
keyfile = os.path.join(path,outpath,out_loc_file)

# Plot filenames:
BJQFileName = 'test_BJQ.png'
GeomFileName = 'test_Geom.png'

# X-axis labels:
XAxisLabel = 'Time on 26 July 2017'

# Desired resolution of data for the curlometer calculation
window = 1/128 # in seconds, 0.2 = minimum FOR CLUSTER

def filenames_get(name_list_file):
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
    """
    cdf_file=cdflib.CDF(filename,varnames)
    data=[]
    for varname in varnames:
        var_data=np.array(cdf_file.varget(varname)) #TRYING NO NP ARRAY (DEBUG)
        data.append(var_data)
    return data

def time_converter(time_nanosecs):
    """
    converts MMS CDF times (nanoseconds, TT 2000) to 
    matplotlib datetime objects
    """    
    #TODO: make sure this thing is working correctly!!!
    start_date=dt.datetime(2000,1,1,hour=11,minute=58,second=55,
                                 microsecond=816000)
        # see https://aa.usno.navy.mil/faq/docs/TT.php for explanation of
        # this conversion time
    start_num=dates.date2num(start_date) #num of start date of time data
    fudge_factor=5/60/60/24 #correction amount (unsure why need)
        #incidentally 5 is the number of leap seconds since 2000. Coincidence?
    time_days=time_nanosecs/1e9/60/60/24 #convert nanoseconds to days
    time_num=time_days+start_num-fudge_factor #convert TT2000 to matplotlib num time
    times=[]
    if time_num.size>1:
        for t in time_num:
            t_dt=dates.num2date(t) #conversion to datetime object
            t_utc=t_dt.astimezone(pytz.utc) #officially set timezone to UTC
            times.append(t_utc)
        return times        
    else:
        t_dt=dates.num2date(time_num) #conversion to datetime object
        t_utc=t_dt.astimezone(pytz.utc) #officially set timezone to UTC
        return t_utc


'''The Curlometer Function'''
def delta(ref, i):
    delrefi = i - ref
    return delrefi

def curlometer(d1, d2, d3, d4):
    
    km2m = 1e3
    nT2T = 1e-9
    mu0 = (4*math.pi)*1e-7
    
    C1R = np.array([d1[3], d1[4], d1[5]])*km2m
    C1B = np.array([d1[0], d1[1], d1[2]])*nT2T
    C2R = np.array([d2[3], d2[4], d2[5]])*km2m
    C2B = np.array([d2[0], d2[1], d2[2]])*nT2T
    C3R = np.array([d3[3], d3[4], d3[5]])*km2m
    C3B = np.array([d3[0], d3[1], d3[2]])*nT2T
    C4R = np.array([d4[3], d4[4], d4[5]])*km2m
    C4B = np.array([d4[0], d4[1], d4[2]])*nT2T
    
    delB14 = delta(C4B, C1B)
    delB24 = delta(C4B, C2B)
    delB34 = delta(C4B, C3B)
    delR14 = delta(C4R, C1R)
    delR24 = delta(C4R, C2R)
    delR34 = delta(C4R, C3R)

# J

    # Have to 'convert' this to a matrix to be able to get the inverse.
    R = np.matrix(([np.cross(delR14, delR24), np.cross(delR24, delR34),
         np.cross(delR14, delR34)]))
    Rinv = R.I

    # I(average) matrix:
    Iave = ([np.dot(delB14, delR24) - np.dot(delB24, delR14)],
        [np.dot(delB24, delR34) - np.dot(delB34, delR24)],
        [np.dot(delB14, delR34) - np.dot(delB34, delR14)])

    JJ = (Rinv*Iave)/mu0
                  
# div B
    lhs = np.dot(delR14, np.cross(delR24, delR34))

    rhs = np.dot(delB14, np.cross(delR24, delR34)) + \
        np.dot(delB24, np.cross(delR34, delR14)) + \
        np.dot(delB34, np.cross(delR14, delR24))

    divB = abs(rhs)/abs(lhs)

# div B / curl B
    curlB = JJ*mu0
    magcurlB = math.sqrt(curlB[0]**2 + curlB[1]**2 + curlB[2]**2)
    divBbycurlB = divB/magcurlB

    return [JJ, divB, divBbycurlB]
# End of curlometer function


'''Read in all the data using CEFLIB.read '''

MMS_num = [str(x) for x in range(1,5)]
MMS=['MMS'+str(x) for x in range(1,5)]

loc_file_whole=os.path.join(path,loc_file)
bfield_files=filenames_get(loc_file_whole)

#iterating over each time interval    
for file_num,file in enumerate(bfield_files):
    time = {}
    time_pos = {} #because MMS has different time series for position data
    B = {}
    pos = {}
    
    for n,M_num in enumerate(MMS_num):
        file_whole=os.path.join(path,datpath,M_num,"mms"+M_num+file)
        tmp1,tmp2,misshape_1,misshape_2=get_cdf_var(file_whole,
                                                 ['Epoch','Epoch_state',
                                                  'mms'+M_num+'_fgm_b_gsm_brst_l2',
                                                  'mms'+M_num+'_fgm_r_gsm_brst_l2'])
        tmp3=misshape_1.reshape(misshape_1.size//4,4)
        tmp4=misshape_2.reshape(misshape_2.size//4,4)
        time[MMS[n]] = deepcopy(tmp1) #in NANOSECONDS
        time_pos[MMS[n]]=deepcopy(tmp2)
        B[MMS[n]] = deepcopy(tmp3)
        pos[MMS[n]] = deepcopy(tmp4)
    
    '''Align all to MMS1 time coordinates using linear interpolation '''
    clean = {}
    #interpolate all data for MMS 2,3,4 (yes I know it is a mess)
    B_interp=[]
    pos_interp=[]
    for n in range(1,4):
        B_interp.append(interp.interp1d(time[MMS[n]],B[MMS[n]],kind='linear',
                              axis=0,assume_sorted=True))
    for n in range(4):
        pos_interp.append(interp.interp1d(time_pos[MMS[n]],pos[MMS[n]],kind='linear',
                              axis=0,assume_sorted=True))  

    tarr=[] #array of good times (helpful to have outside dict)
    for i,t in enumerate(time[MMS[0]]):
        if ((time[MMS[1]]-t>0).all() or (time[MMS[2]]-t>0).all() or 
            (time[MMS[3]]-t>0).all()):
            #time is outside of bounds of interpolation for at least one craft
            continue
        if ((time[MMS[1]]-t<0).all() or (time[MMS[2]]-t<0).all() or 
            (time[MMS[3]]-t<0).all()):
            #time is outside of bounds of interpolation for at least one craft
            continue  
        clean[t]={}
        tarr.append(t)
        for j,M_str in enumerate(MMS):
            if M_str =="MMS1":
                clean[t][M_str] = [B[M_str][i][0], 
                                   B[M_str][i][1],
                                   B[M_str][i][2],
                                   pos_interp[j](t)[0],
                                   pos_interp[j](t)[1],
                                   pos_interp[j](t)[2]]
            else:
                clean[t][M_str] = [B_interp[j-1](t)[0], 
                                   B_interp[j-1](t)[1],
                                   B_interp[j-1](t)[2],
                                   pos_interp[j](t)[0],
                                   pos_interp[j](t)[1],
                                   pos_interp[j](t)[2]]
                               
    mintime, maxtime = min(clean.keys()), max(clean.keys())
    
    # Time array (min, max, step)
    nwin = len(tarr)
    Jave = np.zeros(nwin, dtype = [('time', float),('Jx', float),
                                      ('Jy', float),('Jz', float),
                                      ('divB', float), 
                                      ('divBcurlB', float)])
    
    for i,t in enumerate(clean):
        if len(clean[t]) == 4:
            onej = curlometer(clean[t]['MMS1'],clean[t]['MMS2'],
                             clean[t]['MMS3'],clean[t]['MMS4'])
    
            Jave['time'][i] = t #in nanosecs
            Jave['Jx'][i] = onej[0][0]
            Jave['Jy'][i] = onej[0][1]
            Jave['Jz'][i] = onej[0][2]
            Jave['divB'][i] = onej[1]
            Jave['divBcurlB'][i] = onej[2]
        else:
            Jave['time'][i] = t
            Jave['Jx'][i] = np.nan
            Jave['Jy'][i] = np.nan
            Jave['Jz'][i] = np.nan
            Jave['divB'][i] = np.nan
            Jave['divBcurlB'][i] = np.nan
    
    '''Write all results out to file, tarr is already sorted'''
    
    with open(outfile+str(file_num)+".txt", 'w') as f:
        for j in Jave:
            t_dt=mt.TTtime2datetime(j['time'])
            time_string=mt.datetime2str(t_dt)
            outstring = "{},{},{},{},{}\n".format(time_string,j['Jx'],
                         j['Jy'],j['Jz'],j['divBcurlB'])
            f.write(outstring)

''' Write out what the output files are to the output key file '''
with open(keyfile, 'w') as f:
    for file_num,file in enumerate(bfield_files):
        outstring=outfile+str(file_num)+".txt"+'\n'
        f.write(outstring)

#check how long the code took to run
end=tm.time()
print("Code executed in "+str(dt.timedelta(seconds=end-start)))        

#TODO: Possibly try different interpolation styles than linear?
        