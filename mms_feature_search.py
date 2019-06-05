# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:41:44 2019

@author: kbergste
"""

"""
A current sheet / flux rope searcher for MMS burst data.
"""
#mms-specific in-house modules
import mmsplotting as mmsp

#canned packages
import numpy as np
import cdflib #NEEDS python 3.7 to run
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import pytz #for timezone info
import sys #for debugging
#from pympler.tracker import SummaryTracker #for tracking memory usage
from dateutil.parser import parse #for reading dates from file
import scipy.interpolate as interp #maybe?? For finding crossings???
import scipy.signal as signal #for extrema finding
import scipy.constants as const
import time #for checking code runspeed
import os #for generalization to all systems

#tracker=SummaryTracker() #for tracking memory usage
start=time.time() 

#path and file information
path=os.getcwd()
data_dir="MMS"
bfield_names_file=r"Bfield_file_location_key.txt"
dis_names_file=r"dis_file_location_key.txt"
des_names_file=r"des_file_location_key.txt"
curlometer_dir="Curlometer_data"
j_names_file=os.path.join(curlometer_dir,"curlometer_files.txt")
plot_out_directory=r"feature_search"
plot_out_name=r"Crossing"
statistics_out_directory=os.path.join(plot_out_directory,"statistics")
scales_out_directory=os.path.join(plot_out_directory,
                                  "structure_scale_comparisons")

#parameters (to fiddle with)
boxcar_width=30 #number of points to boxcar average the electron density over
ne_fudge_factor=0.001 #small amount of density to add to avoid NaN velocities
window_padding=20 #number of indices to add to each side of window
extrema_width=20 #number of points to compare on each side to declare an extrema
min_width=15 #minimum number of points between crossings to "count"
min_crossing_height=0.2 #expected nT error in region of interest as per documentation
data_gap_time=dt.timedelta(milliseconds=10) #amount of time to classify 
                                                  #something as a data gap
quality_min=0.5 #used in structure_classification, minimum accepted quality
nbins=20 #number of bins for histograms
window_scale_factor=10  #amount to scale window by for scale comparisons
                                                  
#constants (probably shouldn't change)                                                  
E_CHARGE_mC=const.e*1e6 #electron charge in microcoulombs
REPLOT=1 #chooses whether to regenerate the graphs or not


    
def directory_ensurer(directory):
    '''
    Ensures that the directory specified by the given path exists
    Important for initializing the code on a new machine, since folders
    that only contain output files which are not tracked by git (.png)
    will not be initially present in the github file
    Inputs:
        directory- string containing full path to the desired directory
    Outputs:
        none
    '''
    os.makedirs(directory, exist_ok=True)

    
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

def TTtime2datetime(time_nanosecs):
    """
    converts MMS CDF times (nanoseconds, TT 2000) to 
    matplotlib datetime objects
    """    
    start_date=dt.datetime(2000,1,1,hour=11,minute=58,second=55,
                                 microsecond=816000)
        # see https://aa.usno.navy.mil/faq/docs/TT.php for explanation of
        # this conversion time
    start_num=mpl.dates.date2num(start_date) #num of start date of time data
    fudge_factor=5/60/60/24 #correction amount (unsure why need)
        #incidentally 5 is the number of leap seconds since 2000. Coincidence?
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
    '''
    start_date=dt.datetime(2000,1,1,hour=11,minute=58,second=55,
                                 microsecond=816000)
    start_num=mpl.dates.date2num(start_date)
    fudge_factor=5/60/60/24
    
    time_num=mpl.dates.date2num(time_dt)
    time_days=time_num-start_num+fudge_factor
    time_nanosecs=time_days*24*60*60*1e9
    return time_nanosecs
    
    
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
        var_data=np.array(cdf_file.varget(varname))
        data.append(var_data)
    return data

def import_jdata(filename):
    '''
    Imports current density data outputted from the mms_curlometer script.
    Format is time (string format),jx,jy,jz
    '''
    amps_2_uamps=1e6
    time_str=np.loadtxt(filename,delimiter=',',usecols=[0],dtype="str")
    j_data=np.loadtxt(filename,delimiter=',',usecols=range(1,4))*amps_2_uamps
    time_clean=[]
    for t in time_str:
        time_clean.append(parse(t))

    return time_clean,j_data

def boxcar_avg(array):
    '''
    Does boxcar averaging for an array over the number of data points
    boxcar_width (global parameter above)
    '''
    weights=np.full(boxcar_width,1.0)/boxcar_width
    boxcar_avg=np.convolve(array,weights,mode='same')
    return boxcar_avg

def electron_veloc_x(j,time_j,vi,ni,time_ni,ne,time_ne):
    '''
    Used to calculate the x component of the electron velocity from curlometer 
    current, ion velocity, and electron and ion densities.
    Only invoked once, but I sectioned it off so it's easier to find/edit
    '''
    vi_interp=interp.interp1d(time_ni,vi[:,0],kind='linear',
                              assume_sorted=True)
    ni_interp=interp.interp1d(time_ni,ni, kind='linear',
                              assume_sorted=True)
    ne_interp=interp.interp1d(time_ne,ne, kind='linear',
                              assume_sorted=True)
    vex_list=[]
#    vex_list_novi=[]
    for n,ttime in enumerate(time_j):
        vex_time=(vi_interp(ttime)*ni_interp(ttime)*1e9* \
                  E_CHARGE_mC-j[n,0])/E_CHARGE_mC/(ne_interp(ttime))/1e9
#        vex_time_novi=-j_curl[n,0]/E_CHARGE_mC/ne_interp(time)/1e9
        vex_list.append(vex_time)
#        vex_list_novi.append(vex_time_novi)    
    return np.array(vex_list)

def plasma_frequency(density,mass,zeff=1):
    '''
    defines the plasma frequency in rad/s given density and mass
    mass expected in kg
    density expected in cm^-3
    '''
    charge=zeff*const.e
    density_m3=density*1e6 #cm^-3 to m^-3
    freq=np.sqrt(density_m3*charge*charge/mass/const.epsilon_0)
    print("plasma frequency")
    print(freq)
    return freq

def inertial_length(freq):
    '''
    defines the inertial length in km given the plasma frequency in rad/s
    for either electrons or some ion species
    '''
    c_km_s=const.c/1e3 #m/s to km/s
    d=c_km_s/freq
    print("inertial length")
    print(d)
    return d

def section_maker(indices,maxes,mins,max_index,min_index=0):
    '''
    Takes a tuple of indices of instances and makes a suitably-sized index
    window around each instance using the surrounding extrema, for plotting.
    Zoom in on the crossing itself
    Possibility of missing larger structure
    '''
    window_list=[]
    mins_arr=np.asarray(mins)[0,:] #tuple to numpy array for using searches
    maxes_arr=np.asarray(maxes)[0,:]

    for n,current_index in enumerate(indices):
        min_idx=np.searchsorted(mins_arr,current_index,side='left')
        max_idx=np.searchsorted(maxes_arr,current_index,side='left')
        index_min=min(maxes_arr[max_idx-1],mins_arr[min_idx-1])-window_padding
        index_max=max(maxes_arr[max_idx],mins_arr[min_idx])+window_padding
        window_list.append([index_min,index_max])
    return window_list

def larger_section_maker(window,max_index):
    '''
    For select events, creates a window which is larger than the given window 
    by the window scale factor
    '''
    half_width=int((window[1]-window[0])/2)
    min_window=max(window[0]-window_scale_factor*half_width,0)
    max_window=min(window[1]+window_scale_factor*half_width,max_index)
    return [min_window,max_window]

def interval_mask(series,min_val,max_val):
    '''
    Takes numpy array data and returns a mask defined by given
    minimum and maximum values
    ''' 
    mask=np.logical_and(series > min_val,series < max_val)
    return mask    

def find_crossings(array,timeseries):
    '''
    Finds indices of zero crossings for an array
    
    Also need to screen out crossings that happen in data gaps
    Could modify to find crossings for any value- not useful at the moment
    '''
    arr_sign=np.sign(array)
    arr_compare=abs(arr_sign[:-1]-arr_sign[1:]) #compares each element to next one
    arr_crossing=(arr_compare == 2) #true if there is a crossing
    indices_rough=np.nonzero(arr_crossing) #indices with crossings to their
                                           #immediate right
    cleaned_indices=[]
    for n,index in enumerate(indices_rough[0]):
        if (timeseries[index+1]-timeseries[index] < data_gap_time) :
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

def find_maxes_mins(array,indices,directions):
    '''
    Given list of zero crossings, finds maxes or mins between crossings
    Clean out crossings that don't attain at least 
    the minimum allowed crossing height
    Also make maxes/mins list for defining the size of the windows/structures
    '''
    cleaned_indices_arr=np.array(indices) #to np array for processing purposes
    #find indices of mins and maxes for later window/structure processing
    max_indices=signal.argrelmax(array,order=extrema_width)
    min_indices=signal.argrelmin(array,order=extrema_width)
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
            else: #last index
                next_ext=np.amax(array[indices[n]:indices[n]+50]) 
                prev_ext=np.amin(array[indices[n-1]:indices[n]])
        else: #crossing from pos to neg
            if (n < len(indices)-1) and n>0 : #index on each side
                next_ext=np.amin(array[indices[n]:indices[n+1]])
                prev_ext=np.amax(array[indices[n-1]:indices[n]])
            elif (n < len(indices)-1): #first index
                next_ext=np.amin(array[indices[n]:indices[n+1]])
                prev_ext=np.amax(array[indices[n]-50:indices[n]])
            else: #last index
                next_ext=np.amin(array[indices[n]:indices[n]+50]) 
                prev_ext=np.amax(array[indices[n-1]:indices[n]])
        prev_exts=np.append(prev_exts,[prev_ext]) #add to list
        next_exts=np.append(next_exts,[next_ext]) #add to list

    #Implement (complicated, inefficient) selection mechanism
    #will probably need to be explained in words (or flowchart)
    mask=np.array([],dtype=bool)
    for n,(direc,prev_ext,next_ext) in enumerate(zip(directions,
                                                   prev_exts,next_exts)):
        if (abs(prev_ext)<min_crossing_height):
            mask=np.append(mask,[False])
        elif (abs(prev_ext)>=min_crossing_height and 
            abs(next_ext)>=min_crossing_height):
            mask=np.append(mask,[True])
        else:
            i=n+1
            while i<len(directions):
                if (abs(next_exts[i])>=min_crossing_height and 
                    directions[i]==direc): #trend is continuing up or down across 0
                    mask=np.append(mask,[True])
                    break
                elif (abs(next_exts[i])>=min_crossing_height): #up then down or vice versa
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
    '''
    size_list=[]
    times_list=[]
    mins_arr=np.asarray(mins)[0,:] #tuple to numpy array for using searches
    maxes_arr=np.asarray(maxes)[0,:]

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

def find_avg_signs(array):
    '''
    Given an array of floats, determines the average sign of the data
    as positive or negative
    Also returns an indication of how good the calculation was
    1 is good, 0 is godawful
    ''' 
    total=np.nansum(array) #treat NaNs as zero
    abs_total=np.nansum(np.abs(array)) #treat NaNs as zero
    sign=np.sign(total)
    quality=abs(total/abs_total)
    return sign,quality

def structure_classification(crossing_direction,vex_direction,vex_quality,
                             jy_direction,jy_quality):
    '''
    Determines the tentative kind of structure using:
    -The direction of the Bz crossing
    -The sign of vex
    -The sign of jy
    If the determination of the signs of vex or jy are not of a certain quality
    The classification reteurns 'uncertain'
    '''
    if((jy_quality < quality_min) or (vex_quality < quality_min)):
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
    endtimes: list of start and stop time
    velocs: numpy array of velocities in km/s
    
    uses rms velocity to avoid complications of the velocity changing sign
    
    returns size in km
    '''    
    time_interval=(endtimes[1]-endtimes[0]).total_seconds()
    speed_avg=abs(np.average(velocs))
    
    return speed_avg*time_interval

def fluct_abt_avg(array):
    '''
    Returns the array minus its average
    To show how it fluctuates
    '''
    return array - np.average(array)

###### MAIN ###################################################################
#ensuring that the needed output directories exist
directory_ensurer(os.path.join(path,plot_out_directory))
directory_ensurer(os.path.join(path,statistics_out_directory))
directory_ensurer(os.path.join(path,scales_out_directory))

#initialize variables that cannot be local to loop over MMS satellites
MMS=[str(x) for x in range(1,5)]
MMS_structure_counts={} #dictionary for counts of each structure type
MMS_allstruct_sizes={}
MMS_plasmoid_sizes={}
MMS_cs_sizes={}
MMS_merging_cs_sizes={}

j_curl=np.transpose(np.array([[],[],[]]))  #for all j data
time_reg_jcurl=np.array([])

#repeating for each satellite
for M in MMS:
    MMS_structure_counts[M]={'plasmoid': 0,'pull cs': 0,
                             'push cs': 0,
                             'unclear case': 0,
                             'matches none': 0}
    MMS_allstruct_sizes[M]=np.array([])
    MMS_plasmoid_sizes[M]=np.array([])
    MMS_cs_sizes[M]=np.array([])
    MMS_merging_cs_sizes[M]=np.array([])
    
    b_list=filenames_get(os.path.join(path,data_dir,M,bfield_names_file))
    dis_list=filenames_get(os.path.join(path,data_dir,M,dis_names_file))
    des_list=filenames_get(os.path.join(path,data_dir,M,des_names_file))
    j_list=filenames_get(os.path.join(path,j_names_file))

    b_labels=['MMS'+M+' GSM B-field vs. time', 'Time','Bz GSM (nT)']
    j_labels=['MMS GSM curlometer current density vs. time','Time', 
              r'Jy GSM (microA/m^2)'  ]
    v_labels=['MMS GSM velocities vs. time','Time',
               r'Vx GSM (km/s)']
    v_legend=['$v_e$ from curlometer','$v_e$ rom moments data',
               '$v_i$ from moments data']
    n_labels=['MMS'+M+' density vs. time','Time','density cm^(-3)']
    n_legend=['ion density','electron density','smoothed electron density',
              'ion density error','electron density error']
    fluct_labels=['MMS'+M+' velocity fluctuations vs. time','Time',
                  r'Vx fluctuations GSM (km/s)']
    fluct_legend=[r'$v_i$ fluctuations from moments data',
                  r'$v_e$ fluctuations from curlometer']
    
    b_field=np.transpose(np.array([[],[],[],[]])) #for all B field data
    ni=np.array([])
    ni_err=np.array([])
    ne=np.array([])
    ne_err=np.array([])
    vi=np.transpose(np.array([[],[],[]])) #for all vi data      
    ve_fpi=np.transpose(np.array([[],[],[]])) #for ve from MMS data directly
    TT_time_b=np.array([])
    TT_time_ni=np.array([])
    TT_time_ne=np.array([])
    
    for b_stub,j_stub,dis_stub,des_stub in zip(b_list,j_list,dis_list,
                                               des_list):
        #create full paths
        b_file=os.path.join(path,data_dir,M,b_stub)
        j_file=os.path.join(path,data_dir,M,j_stub)
        dis_file=os.path.join(path,data_dir,M,dis_stub)
        des_file=os.path.join(path,data_dir,M,des_stub)
        #read and process b-field data
        TT_time_tmp,temp=get_cdf_var(b_file,['Epoch',
                                             'mms'+M+'_fgm_b_gsm_brst_l2'])
        b_field_tmp=temp.reshape(temp.size//4,4) #// returns integer output
        b_field=np.concatenate((b_field,b_field_tmp),axis=0)
        TT_time_b=np.concatenate((TT_time_b,TT_time_tmp))
        #read and process the ni,vi data
        TT_time_tmp,ni_tmp,ni_err_tmp,temp=get_cdf_var(dis_file,
                                       ['Epoch',
                                        'mms'+M+'_dis_numberdensity_brst',
                                        'mms'+M+'_dis_numberdensity_err_brst',
                                        'mms'+M+'_dis_bulkv_gse_brst'])
        ni=np.concatenate((ni,ni_tmp))
        ni_err=np.concatenate((ni_err,ni_err_tmp))
        TT_time_ni=np.concatenate((TT_time_ni,TT_time_tmp))
        vi_tmp=temp.reshape(temp.size//3,3)
        vi=np.concatenate((vi,vi_tmp))
        #read and process the ne and ve data
        TT_time_tmp,ne_tmp,ne_err_tmp,temp=get_cdf_var(des_file,
                                       ['Epoch',
                                        'mms'+M+'_des_numberdensity_brst',
                                        'mms'+M+'_des_numberdensity_err_brst',
                                        'mms'+M+'_des_bulkv_gse_brst'])
        ne=np.concatenate((ne,ne_tmp))
        ne_err=np.concatenate((ne_err,ne_err_tmp)) #note errors
        TT_time_ne=np.concatenate((TT_time_ne,TT_time_tmp)) 
        ve_tmp=temp.reshape(temp.size//3,3)
        ve_fpi=np.concatenate((ve_fpi,ve_tmp))
        #read and process the j data
        if (M=='1'):
            tmp,j_curl_tmp=import_jdata(j_file)
            time_reg_j_tmp=np.array(tmp)
            j_curl=np.concatenate((j_curl,j_curl_tmp),axis=0)
            time_reg_jcurl=np.concatenate((time_reg_jcurl,time_reg_j_tmp))
    TT_time_j=datetime2TTtime(time_reg_jcurl) #time to nanosecs for interpolating
    time_reg_b=np.array(TTtime2datetime(TT_time_b)) #time as datetime obj np arr
    time_reg_ni=np.array(TTtime2datetime(TT_time_ni))
    time_reg_ne=np.array(TTtime2datetime(TT_time_ne))
    bz=b_field[:,2]
    jy=j_curl[:,1]
    vex_fpi=ve_fpi[:,0]
    vix=vi[:,0]
    ne_smooth=boxcar_avg(ne) #smooth the ne data to avoid zeroes
    ne_nozero=np.where(ne_smooth>ne_fudge_factor,ne_smooth,ne_fudge_factor)
    #roughly calculate electron velocity from curlometer
    vex=electron_veloc_x(j_curl,TT_time_j,vi,ni,TT_time_ni,ne_nozero,
                         TT_time_ne)
    #calculate approximate electron and ion plasma frequencies and skin depths
    we=plasma_frequency(ne_nozero,const.m_e)
    wp=plasma_frequency(ni,const.m_p) #assuing all ions are protons (valid?)
    de=inertial_length(we)
    dp=inertial_length(wp)
    
    #locate crossings and their directions
    crossing_indices_bz=find_crossings(bz,time_reg_b)
    crossing_signs_bz=find_crossing_signs(bz,crossing_indices_bz)
    crossing_indices_bz,crossing_signs_bz,max_indices,min_indices= \
                                                        find_maxes_mins(bz,
                                                        crossing_indices_bz,
                                                        crossing_signs_bz)
    crossing_times=time_reg_b[crossing_indices_bz]
    #section the data and define structural extents
    crossing_windows=section_maker(crossing_indices_bz,max_indices,min_indices,
                                   len(bz))
    crossing_structs,crossing_struct_times=structure_extent(
                                    crossing_indices_bz,time_reg_b,
                                    crossing_signs_bz,max_indices,min_indices,
                                    len(bz))
    #process each crossing
    for i in range(len(crossing_indices_bz)):
        #slice b and b timeseries, set plotting limits
        time_b_cut=time_reg_b[crossing_windows[i][0]:crossing_windows[i][1]]
        bz_cut=bz[crossing_windows[i][0]:crossing_windows[i][1]] #for window
        time_b_struct=time_reg_b[crossing_structs[i][0]:crossing_structs[i][1]]
        plot_limits=[time_b_cut[0],time_b_cut[-1]] #data section
        #slice ni,vi and ni timeseries
        window_mask_ni=interval_mask(time_reg_ni,plot_limits[0],plot_limits[1])
        struct_mask_ni=interval_mask(time_reg_ni,time_b_struct[0],
                                     time_b_struct[1])
        time_ni_cut=time_reg_ni[window_mask_ni]
        ni_cut=ni[window_mask_ni]
        ni_err_cut=ni_err[window_mask_ni]
        vix_cut=vix[window_mask_ni]
        vix_fluct=fluct_abt_avg(vix_cut) #for fluctuation plot
        #slice ne and ne timeseries
        window_mask_ne=interval_mask(time_reg_ne,plot_limits[0],plot_limits[1])
        struct_mask_ne=interval_mask(time_reg_ne,time_b_struct[0],
                                     time_b_struct[1])
        time_ne_cut=time_reg_ne[window_mask_ne]
        ne_cut=ne[window_mask_ne]
        ne_smooth_cut=ne_smooth[window_mask_ne]
        ne_err_cut=ne_err[window_mask_ne]  
        #slice ve from FPI timeseries
        vex_fpi_cut=vex_fpi[window_mask_ne]
        #slice j and j timeseries
        window_mask_j=interval_mask(time_reg_jcurl,plot_limits[0],
                                    plot_limits[1])
        struct_mask_j=interval_mask(time_reg_jcurl,time_b_struct[0],
                                    time_b_struct[-1])
        time_j_cut=time_reg_jcurl[window_mask_j] #for window
        jy_cut=jy[window_mask_j] #for window
        time_j_struct=time_reg_jcurl[struct_mask_j] #for structure
        jy_struct=jy[struct_mask_j] #for structure
        #slice ve curlometer timeseries (same as j)
        vex_cut=vex[window_mask_j] #for window
        vex_struct=vex[struct_mask_j] #for structure
        vex_fluct=fluct_abt_avg(vex_struct) #for fluctuation plot
        #slice inertial lengths and average
        de_cut=de[window_mask_ne]
        de_cut_avg=np.average(de_cut)
        dp_cut=dp[window_mask_ni]
        dp_cut_avg=np.average(dp_cut)
        str_de_avg=f"{de_cut_avg:.1f}"  #string formatting
        str_dp_avg=f"{dp_cut_avg:.1f}"  #string formatting
        #determine signs of vex and jy
        jy_sign,jy_qual=find_avg_signs(jy_struct)
        vex_sign,vex_qual=find_avg_signs(vex_struct)
        #determine crossing clasification and size and update counts:
        crossing_type,type_flag=structure_classification(crossing_signs_bz[i],
                                                         vex_sign,vex_qual,
                                                         jy_sign,jy_qual)
        crossing_size=structure_sizer([time_b_struct[0],time_b_struct[1]],
                                      vex_struct)
        str_crossing_size=f"{crossing_size:.1f}"  #string formatting
        MMS_allstruct_sizes[M]=np.append(MMS_allstruct_sizes[M],
                                         [crossing_size])
        if (type_flag == 0):
            MMS_structure_counts[M]['plasmoid'] += 1
            MMS_plasmoid_sizes[M]=np.append(MMS_plasmoid_sizes[M],
                                            [crossing_size])
        elif (type_flag == 1):
            MMS_structure_counts[M]['pull cs'] += 1
            MMS_cs_sizes[M]=np.append(MMS_cs_sizes[M],[crossing_size])
        elif (type_flag == 2):
            MMS_structure_counts[M]['push cs'] += 1
            MMS_merging_cs_sizes[M]=np.append(MMS_merging_cs_sizes[M],
                                              [crossing_size])
        elif (type_flag == 3):
            MMS_structure_counts[M]['unclear case'] += 1
        else: #type_flag ==4
            MMS_structure_counts[M]['matches none'] += 1
            
        #plot everything, if desired:
        if (REPLOT):
            jy_sign_label="jy sign is "+str(jy_sign)+" with quality "+ \
            str(jy_qual)+"\n"
            vex_sign_label="vex sign is "+str(vex_sign)+" with quality " \
                            +str(vex_qual)+"\n"
            crossing_sign_label="Crossing type: "+crossing_type+"\n"
            crossing_size_label="Crossing size: "+str_crossing_size+" km"+"\n"
            crossing_de_label="Average electron inertial length: "+ \
                                str_de_avg+" km"+"\n"
            crossing_dp_label="Average proton inertial length: "+ \
                                str_dp_avg+" km"
                
            #plot everything
            mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
            plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
            gridsize=(6,1)
            fig=plt.figure(figsize=(8,15)) #width,height
            ax1=plt.subplot2grid(gridsize,(0,0))
            ax2=plt.subplot2grid(gridsize,(1,0))   
            ax3=plt.subplot2grid(gridsize,(2,0)) 
            ax4=plt.subplot2grid(gridsize,(3,0))
            ax5=plt.subplot2grid(gridsize,(4,0))
            ax6=plt.subplot2grid(gridsize,(5,0))
            ax6.axis('off')
            mmsp.tseries_plotter(fig,ax1,time_b_cut,bz_cut,b_labels,plot_limits) #plot B
            mmsp.tseries_plotter(fig,ax2,time_j_cut,jy_cut,j_labels,plot_limits) #plot jy
            mmsp.tseries_plotter(fig,ax3,time_ni_cut,ni_cut,n_labels,plot_limits,
                          legend=n_legend[0]) #plot ni
            mmsp.tseries_plotter(fig,ax3,time_ne_cut,ne_cut,n_labels,plot_limits,
                          legend=n_legend[1]) #plot ne (same axis)
            mmsp.tseries_plotter(fig,ax3,time_ne_cut,ne_smooth_cut,n_labels,
                            plot_limits,legend=n_legend[2]) #plot smoothed ne (same axis)
#            tseries_plotter(fig,ax3,time_ni_cut,ni_err_cut,n_labels,
#                            plot_limits,legend=n_legend[3]) #plot ni error for comparison
#            tseries_plotter(fig,ax3,time_ne_cut,ne_err_cut,n_labels,
#                            plot_limits,legend=n_legend[4]) #plot ne error for comparison
            mmsp.tseries_plotter(fig,ax4,time_j_cut,vex_cut,v_labels,plot_limits,
                          legend=v_legend[0])#plt vex from curlometer
            mmsp.tseries_plotter(fig,ax4,time_ne_cut,vex_fpi_cut,v_labels,
                            plot_limits,legend=v_legend[1])#plot vex from moments
            mmsp.tseries_plotter(fig,ax4,time_ni_cut,vix_cut,v_labels,plot_limits,
                          legend=v_legend[2])#plot vi from moments    
            mmsp.tseries_plotter(fig,ax5,time_ni_cut,vix_fluct,fluct_labels,
                            plot_limits,legend=fluct_legend[0])#plot vi fluctuations  
            mmsp.tseries_plotter(fig,ax5,time_j_struct,vex_fluct,fluct_labels,
                            plot_limits,legend=fluct_legend[1])#plot ve fluctuations 
            #add horizontal and vertical lines to plot (crossing + extent)
            mmsp.line_maker([ax1,ax2,ax3,ax4,ax5],crossing_times[i],
                       crossing_struct_times[i])
            #add categorization information to plot
            ax6.text(0.5,0.5,jy_sign_label+vex_sign_label+crossing_sign_label \
                     +crossing_size_label+crossing_de_label+crossing_dp_label,
                     wrap=True,transform=ax6.transAxes,fontsize=16,ha='center',
                     va='center')
            fig.savefig(os.path.join(path,plot_out_directory,'MMS'+M+'_'+ \
                        plot_out_name+str(i)+".png"), bbox_inches='tight')
            plt.close(fig='all')
        
            if (i % 30 == 0):
                ''' Plot larger window to check for larger structures '''
                larger_window=larger_section_maker(crossing_windows[i],len(bz))   
                time_b_large=time_reg_b[larger_window[0]:larger_window[1]]
                plot_limits_large=[time_b_large[0],time_b_large[-1]]
                bz_large=bz[larger_window[0]:larger_window[1]] #for window
                
                #plot comparison of structures
                gridsize=(2,1)
                fig=plt.figure(figsize=(8,8)) #width,height
                ax1=plt.subplot2grid(gridsize,(0,0))
                ax2=plt.subplot2grid(gridsize,(1,0))   
                mmsp.tseries_plotter(fig,ax1,time_b_cut,bz_cut,b_labels,plot_limits) #plot B
                mmsp.tseries_plotter(fig,ax2,time_b_large,bz_large,
                                b_labels,plot_limits_large) #plot B large
                mmsp.line_maker([ax1,ax2],crossing_times[i],
                           crossing_struct_times[i])   
                fig.savefig(os.path.join(path,scales_out_directory,'MMS'+M+'_'+ \
                            plot_out_name+str(i)+".png"), bbox_inches='tight')
                plt.close(fig='all')
            
#        tracker.print_diff() #for detecting memory leaks

#        if (i==15): #debug option
#            #check how long the code took to run
#            end=time.time()
#            print("Code executed in "+str(dt.timedelta(seconds=end-start)))   
#            sys.exit("done with test cases")  

""" STATISTICAL PART """

''' make bar chart of the different types of structures '''
fig_bar,ax_bar=plt.subplots()
mmsp.bar_charter(ax_bar,MMS_structure_counts,['Types of structures seen by MMS',
                                         'Type of structure',
                                         'Number of instances']) 
fig_bar.savefig(os.path.join(path,statistics_out_directory,
                             "types_bar_chart"+".png"),bbox_inches='tight')
plt.close(fig='all')
''' make histograms of the x-lengths of all structures''' 
hist_path=os.path.join(path,statistics_out_directory)
mmsp.structure_hist_maker(MMS_allstruct_sizes,"all structures",hist_path,nbins)
mmsp.structure_hist_maker(MMS_plasmoid_sizes,"plasmoids",hist_path,nbins)
mmsp.structure_hist_maker(MMS_cs_sizes,"pull current sheets",hist_path,nbins)
mmsp.structure_hist_maker(MMS_merging_cs_sizes,"push current sheets",hist_path,
                     nbins)
''' make histograms on log scale of the x-lengths of all structures '''
mmsp.structure_hist_maker(MMS_allstruct_sizes,"all structures",hist_path,nbins,
                     log=True)
mmsp.structure_hist_maker(MMS_plasmoid_sizes,"plasmoids",hist_path,nbins,log=True)
mmsp.structure_hist_maker(MMS_cs_sizes,"pull current sheets",hist_path,nbins,
                     log=True)
mmsp.structure_hist_maker(MMS_merging_cs_sizes,"push current sheets",hist_path,
                     nbins,log=True)               
                
#check how long the code took to run
end=time.time()
print("Code executed in "+str(dt.timedelta(seconds=end-start)))    
    
    
        
            

#Urgent Priorities:
#TODO: make rudimentary estimate of plasma frequencies and length scales
       #plot in log scale?
#TODO: normalize by length scales? Maybe just for printouts
        #that would be more easily doable
#TODO: change structure extent determination, possibly using a sliding scale?
        #must reach this distance unless the next crossing is closer?
        #possibly wait on this and discuss with group
#TODO: try to fit power law / exponential?
        #maybe wait and discuss potential fitting (ideas from literature)
#TODO: read a lot of the literature!!!
        #plasmoid statistics studies, waves in the magnetotail, etc.
#TODO: think about involving data from the search coil magnetometer? 
        #could help screen out waves?
        
#Later Priorities:
#TODO: clean up functions indo modules
#TODO: make sure function documentation (inputs,outputs, etc.) is clear
#TODO: pass all 'global' parameters to functions so they can be shoved in
        #modules without issues
#TODO: interpolate Bz to find exact time of zero crossing  for vertical line
#TODO: capitalize all parameters
#TODO: set maximum yrange of the velocity data to max of curlometer ve  
#TODO: adapt code to conform to PEP-8 standards
#TODO: attempt to fit a power law to the data 
#TODO: Keep window sizes from getting extreme (see MMS4 structure 1...)
#TODO: once can plot all for all spacecraft, figure out how to determine 
        #which structures found are the same
#TODO: further work on making electron velocity calculation robust
        #using more or less averaged/different density maybe
        #maybe shifting the curlometer current density to account for 
        #spatial separation of satellites
        #maybe delving into the electron phase space densities to redo the 
            #moment calculation
#TODO: gain understanding of how the fpi instruments work and what their 
        #limitations are.
#TODO: check if limiting local maxes/mins to greater than min crossing height 
        #is better for window/structure size making or not
        
'''
MISC THOUGHTS:
For a current sheet (not from merging plasmoids),
the satellites won't necessarily cross the current sheet to see a Bz change
and thus might not see an enhancement of jy (though it might).

Need to do some reading into the kind of waves observable in this part of the
magnetotail- want to be able to implement measures (either thru code or by eye)
to avoid tagging wave-related time-dependent crossings with spatial structures
Concern: solitons??? Ion cyclotron? etc.
Does the concept of a 'wave' even make sense in a turbulent plasma?
If not, problem becomes instead determining which perturbations are time-based
'''
        