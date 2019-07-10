# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 11:43:00 2019

Script to do multi-spacecraft analysis (to supplement the work done by 
mms_feature_search)
Hope is to determine accurate structure speeds, invariant directions, etc.
@author: kbergste
"""

#mms-specific in-house modules
import mmsplotting as mmsp
import mmstimes as mt
import plasmaparams as pp
import mmsdata as md
import mmsarrays as ma
import mmsstructs as ms
import mmsmultispacecraft as msc

#canned packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import sys #for debugging
#from pympler.tracker import SummaryTracker #for tracking memory usage
import scipy.constants as const
import scipy.interpolate as interp
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
plot_out_directory=r"ms_feature_search"
plot_out_name=r"ms_Crossing"
timeseries_out_directory=os.path.join(path,plot_out_directory)

#parameters
data_gap_time=dt.timedelta(milliseconds=10) #amount of time to classify 
                                                  #something as a data gap
extrema_width=10 #number of points to compare on each side to declare an extrema
min_crossing_height=0.1 #expected nT error in region of interest as per documentation

DEBUG=1 #chooses whether to stop at iteration 15 or not

###### CLASS DEFINITIONS ######################################################
class Structure:
    
    #initializer
    def __init__(self,kind,size,gf):
        self.kind=kind
        self.size=size
        self.guide_field=gf
        
    #for pluralization in outputs
    plurals={'size':'sizes',
                  'kind':'kinds',
                  'guide_field':'guide fields',
                }
    
    #for getting units
    units={ 'size':'km',
                'kind':'',
                'guide_field':'nT',
            }

###### MAIN ###################################################################
#ensuring that the needed output directories exist
mmsp.directory_ensurer(timeseries_out_directory)

#initialize variables, dictionaries, etc.
MMS=[str(x) for x in range(1,5)]
b_field={} #dictionary that will contain all the different magnetic field data
rad={} #dictionary holding resampled spacecraft position data (same time cadence as that satellite's b-field)
TT_time_b={}
time_reg_b={}
ni={}
vi={}
TT_time_ni={}

j_curl=np.transpose(np.array([[],[],[]]))  #for all j data
time_reg_jcurl=np.array([])

# get data for all satellites
j_list=md.filenames_get(os.path.join(path,j_names_file))
for j_file in j_list:
    time_reg_j_tmp,j_curl_tmp=md.import_jdata(j_file)
    j_curl=np.concatenate((j_curl,j_curl_tmp),axis=0)
    time_reg_jcurl=np.concatenate((time_reg_jcurl,time_reg_j_tmp))
TT_time_j=mt.datetime2TTtime(time_reg_jcurl) #time to nanosecs for interpolating
    
for M in MMS:
    b_list=md.filenames_get(os.path.join(path,data_dir,M,bfield_names_file))
    dis_list=md.filenames_get(os.path.join(path,data_dir,M,dis_names_file))
    des_list=md.filenames_get(os.path.join(path,data_dir,M,des_names_file))
    
    b_field[M]=np.transpose(np.array([[],[],[]])) #for all B field data
    rad[M]=np.transpose(np.array([[],[],[]])) 
    TT_time_b[M]=np.array([])
    ni[M]=np.array([])
    vi[M]=np.transpose(np.array([[],[],[]])) #for all vi data 
    TT_time_ni[M]=np.array([])
    
    for b_stub,dis_stub,des_stub in zip(b_list,dis_list,des_list):
        #create full paths
        b_file=os.path.join(path,data_dir,M,b_stub)     
        dis_file=os.path.join(path,data_dir,M,dis_stub)
        des_file=os.path.join(path,data_dir,M,des_stub)
        #read and process b-field data
        TT_time_tmp,TT_radtime_tmp,temp,temp2=md.get_cdf_var(b_file,['Epoch',
                                                                     'Epoch_state',
                                             'mms'+M+'_fgm_b_gsm_brst_l2',
                                             'mms'+M+'_fgm_r_gsm_brst_l2'])
        b_field_tmp=temp.reshape(temp.size//4,4)[:,0:3] #EXCLUDE the total b-field
        rad_tmp=temp2.reshape(temp2.size//4,4)[:,0:3] #EXCLUDE the total radius
        tmp_rad_spline=interp.CubicSpline(TT_radtime_tmp,rad_tmp) #interpolate position data
        rad_btime_tmp=tmp_rad_spline(TT_time_tmp) #interpolate position to b-field timestamps
        b_field[M]=np.concatenate((b_field[M],b_field_tmp),axis=0)
        rad[M]=np.concatenate((rad[M],rad_btime_tmp),axis=0)
        TT_time_b[M]=np.concatenate((TT_time_b[M],TT_time_tmp))
        #read and process the ni,vi data
        TT_time_tmp,ni_tmp,ni_err_tmp,temp=md.get_cdf_var(dis_file,
                                       ['Epoch',
                                        'mms'+M+'_dis_numberdensity_brst',
                                        'mms'+M+'_dis_numberdensity_err_brst',
                                        'mms'+M+'_dis_bulkv_gse_brst'])
        ni[M]=np.concatenate((ni[M],ni_tmp))
        TT_time_ni[M]=np.concatenate((TT_time_ni[M],TT_time_tmp))
        vi_tmp=temp.reshape(temp.size//3,3)
        vi[M]=np.concatenate((vi[M],vi_tmp))
    
    #populate other necessary data dictionaries
    time_reg_b[M]=np.array(mt.TTtime2datetime(TT_time_b[M])) #time as datetime obj np arr


#find potential structure candidates from MMS 1
bz_M1=b_field[MMS[0]][:,2]
crossing_indices_M1=ms.find_crossings(bz_M1,time_reg_b[MMS[0]],data_gap_time)
crossing_signs_M1=ms.find_crossing_signs(bz_M1,crossing_indices_M1)
crossing_indices_M1,crossing_signs_M1,max_indices_M1,min_indices_M1= \
                                                    ms.find_maxes_mins(bz_M1,
                                                    crossing_indices_M1,
                                                    crossing_signs_M1,
                                                    extrema_width,
                                                    min_crossing_height)
crossing_structs,crossing_struct_times=ms.structure_extent(
                                crossing_indices_M1,time_reg_b[MMS[0]],
                                crossing_signs_M1,max_indices_M1,
                                min_indices_M1,len(bz_M1))
#investigate each potential structure:
for i in range(len(crossing_indices_M1)):
    if (i==16 and DEBUG): #debug option
        #check how long the code took to run
        end=time.time()
        print("Code executed in "+str(dt.timedelta(seconds=end-start)))   
        sys.exit("done with test cases")   
        
    #determine structure sizes for M1
    time_struct_b={}
    b_field_struct={}
    time_struct_b[MMS[0]]=time_reg_b[MMS[0]][crossing_structs[i][0]: \
                                              crossing_structs[i][1]]
    b_field_struct[MMS[0]]=b_field[MMS[0]][crossing_structs[i][0]: \
                                              crossing_structs[i][1]]
    struct_endpts=[time_struct_b[MMS[0]][0],time_struct_b[MMS[0]][-1]]
    
    #determine structures for multispacecraft techniques
    bad_struct=False #True if all 4 spacecraft do not see the structure
    for n in range(1,4): #over MMS 2,3,4
        struct_mask=ma.interval_mask(time_reg_b[MMS[n]],struct_endpts[0],
                             struct_endpts[1])
        b_field_struct[MMS[n]]=b_field[MMS[n]][struct_mask]
        time_struct_b[MMS[n]]=time_reg_b[MMS[n]][struct_mask]
        is_crossing=ms.find_crossings(b_field_struct[MMS[n]][:,2],
                                      time_struct_b[MMS[n]],data_gap_time)
        if len(is_crossing) is 0: #no crossing for this satellite
            bad_struct=True
                
    if(bad_struct): #not able to do multispacecraft techniques
        continue
      
    #plot it (temp feature- just wanna look now)
    fig,(ax1,ax2)=plt.subplots(2)
    print(i)
    for M in MMS:
        bz=b_field_struct[M][:,2]
        mmsp.tseries_plotter(fig,ax1,time_struct_b[M],bz,
                             labels=['Original','',''],
                             lims=[min(time_struct_b[M]),
                                   max(time_struct_b[M])],
                             legend=M)
        
    #sync all B-field data to MMS1 cadence
    b_field_struct_sync={}
    rad_struct_sync={}
    for M in MMS:
        b_field_struct_sync[M]=msc.bartlett_interp(b_field[M],time_reg_b[M],
                                                   time_struct_b[MMS[0]])
        rad_struct_sync[M]=msc.bartlett_interp(rad[M],time_reg_b[M],
                                               time_struct_b[MMS[0]])
        bz=b_field_struct_sync[M][:,2]
        mmsp.tseries_plotter(fig,ax2,time_struct_b[MMS[0]],bz,
                             labels=['Time sync','Time','B (nT)'],
                             lims=[min(time_struct_b[MMS[0]]),
                                   max(time_struct_b[MMS[0]])],
                             legend=M)
    plt.show()  
    plt.close(fig="all")                                   

    #find spatial gradients
    test=msc.spatial_gradient(b_field_struct_sync,rad_struct_sync)
    sys.exit()    
#check how long the code took to run
end=time.time()
print("Code executed in "+str(dt.timedelta(seconds=end-start))) 

#To-do list:
#TODO: improve mechanism used to determine what time section to do the multi-spacecraft tech on
    #right now just what MMS1 says is the structure   
