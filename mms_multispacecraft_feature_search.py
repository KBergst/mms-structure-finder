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
edp_names_file=r"edp_file_location_key.txt"
curlometer_dir="Curlometer_data"
j_names_file=os.path.join(curlometer_dir,"curlometer_files.txt")
plot_out_directory=r"ms_feature_search"
plot_out_name=r"ms_Crossing"
timeseries_out_directory=os.path.join(path,plot_out_directory)
statistics_out_directory=os.path.join(path,plot_out_directory,"statistics")
hists_out_directory=os.path.join(statistics_out_directory,"hists")
scatters_out_directory=os.path.join(statistics_out_directory,"scatters")

#parameters
data_gap_time=dt.timedelta(milliseconds=10) #amount of time to classify 
                                                  #something as a data gap
extrema_width=10 #number of points to compare on each side to declare an extrema
min_crossing_height=0.1 #expected nT error in region of interest as per documentation
window_padding=20 #number of indices to add to each side of window
ne_fudge_factor=0.001 #small amount of density to add to avoid NaN velocities
quality_min=0.2 #used in structure_classification, minimum accepted quality
nbins_small=10 #number of bins for log-scale histograms and other small hists
nbins=15 #number of bins for the size histograms

DEBUG=0 #chooses whether to stop at iteration 15 or not
REPLOT=0 #chooses whether to regenerate the plots or not
REHIST=1 #chooses whether to regenerate the histograms and scatter plots or not

###### CLASS DEFINITIONS ######################################################
class Structure:
    
    #initializer
    def __init__(self,kind,dt,size,size_e,size_p,cf,gf,vel,vel_pm,vel_e,vel_p,dim,
                 je_para,je_perp):
        self.kind=kind
        self.duration=dt
        self.size=size
        self.electron_normalized_size=size_e
        self.ion_normalized_size=size_p
        self.core_field=cf
        self.guide_field=gf
        self.normal_speed=vel
        self.signed_normal_speed=vel_pm
        self.electron_normalized_speed=vel_e
        self.ion_normalized_speed=vel_p
        self.dimensionality=dim
        self.j_dot_E_parallel=je_para
        self.j_dot_E_perpendicular=je_perp
        
        #back out the ion/electron skin depth because I set this up dumb
        self.ion_skin_depth=size/size_p
        self.electron_skin_depth=size/size_e
        
    #for pluralization in outputs
    plurals={'size':'sizes',
                 'electron_normalized_size':'electron-normalized sizes',
                 'ion_normalized_size':'ion-normalized sizes',
                  'kind':'kinds',
                  'core_field':'core fields',
                  'guide_field':'guide fields',
                  'normal_speed':'normal speeds',
                  'signed_normal_speed':'signed normal speeds',
                  'electron_normalized_speed':'electron-normalized speeds',
                  'ion_normalized_speed':'ion-normalized speeds',
                  'duration':'durations',
                  'dimensionality':'dimensionalities',
                  'j_dot_E_parallel':r'averaged parallel components of J$\cdot$E',
                  'j_dot_E_perpendicular':r'averaged perpendicular components of J$\cdot$E',
                  'electron_skin_depth':r'electron skin depths',
                  'ion_skin_depth':r'ion skin depths'
                }
    
    #for getting units
    units={'size':'km',
                'electron_normalized_size':'electron inertial lengths',
                'ion_normalized_size':'ion inertial lengths',
                'kind':'',
                'core_field':'nT',
                'guide_field':'nT',
                'normal_speed':'km/s',
                'signed_normal_speed':'km/s',
                'electron_normalized_speed':'unitless',
                'ion_normalized_speed':'unitless',   
                'duration':'s',
                'dimensionality':'',
                'j_dot_E_parallel':r'nW/$m^3$',
                'j_dot_E_perpendicular':r'nW/$m^3$',
                'electron_skin_depth':'km',
                'ion_skin_depth':'km',
            }

###### MAIN ###################################################################
#ensuring that the needed output directories exist
mmsp.directory_ensurer(timeseries_out_directory)
mmsp.directory_ensurer(statistics_out_directory)
mmsp.directory_ensurer(hists_out_directory)
mmsp.directory_ensurer(scatters_out_directory)

#initialize variables, dictionaries, etc.
MMS=[str(x) for x in range(1,5)]
MMS_structure_counts={} #dictionary for counts of each structure type
MMS_structures=[] #list of all structures

#for moving from the type_flag to a string categorization
type_dict={
        0:'plasmoids',
        1:'pull current sheets',
        2:'push current sheets',
        3:'unclear cases',
        4:'matches none'
        }

type_strs=["1D structure","2D structure",
           "2D structure with some smaller-dimensional qualities",
           "3D structure",
           "3D structure with some smaller-dimensional qualities"
           ]

b_field={} #dictionary that will contain all the different magnetic field data
rad={} #dictionary holding resampled spacecraft position data (same time cadence as that satellite's b-field)
TT_time_b={} #for magnetic field timeseries in MMS times (TT2000)
time_reg_b={} #for magnetic field timeseries in datetime objects
ni={} #for ion density
vi={} #will be interpolated to b-field cadence (fit w/ spline)
presi={} #ion pressure
ne={} #for electron density
ne_nozero={} #to remove underflows from the electron density data
ve={} #for curlometer electron velocity
prese={} #electron pressure
errflags_i={} #dis errorflags
errflags_e={} #des errorflags
e_field={}

j_curl=np.transpose(np.array([[],[],[]]))  #for all j data
time_reg_jcurl=np.array([])

#variables for global J dot E info
cadence=1/128 #time cadence for MMS burst data
time_tot=0 #at B cadence, average over all 4 satellites
time_structs=0 #at B cadence
time_plasmoid=0
time_pullcs=0
time_pushcs=0
j_dot_E_para_sum={} #average over all 4 satellites
j_dot_E_para_sum['pos']=0.
j_dot_E_para_sum['neg']=0.
j_dot_E_perp_sum=j_dot_E_para_sum.copy() #average over all 4 satellites
j_dot_E_struct_para_sum=j_dot_E_para_sum.copy()
j_dot_E_struct_perp_sum=j_dot_E_para_sum.copy()
j_dot_E_plasmoid_para=j_dot_E_para_sum.copy()
j_dot_E_plasmoid_perp=j_dot_E_para_sum.copy()
j_dot_E_pullcs_para=j_dot_E_para_sum.copy()
j_dot_E_pullcs_perp=j_dot_E_para_sum.copy()
j_dot_E_pushcs_para=j_dot_E_para_sum.copy()
j_dot_E_pushcs_perp=j_dot_E_para_sum.copy()

#legends, etc. for plotting
b_label=['Magnetic field of structure over all satellites','Time','Bz (nT)']
eigenval_label=['Eigenvalues from MDD','Time',r'$\lambda$']
eigenval_legend=[r'$\lambda_{max}$',r'$\lambda_{med}$',r'$\lambda_{min}$']
eigenvec_label_start=['maximum','middle','minimum']
eigenvec_label=[]
for i in range(3):
    eigenvec_label.append(\
      [r'Unit vector components in the {} varying direction' \
       .format(eigenvec_label_start[i]),'Time','Component (GSM)'])
eigenvec_legend=['x','y','z']
veloc_labels=['Structure normal velocity in GSM in time','Time',
              'Normal velocity (km/s)']
veloc_legend=['Vx','Vy','Vz','Vtot']
vcompare_labels=['Velocity at barycenter','Time','Normal velocity (km/s)']
vcompare_legend=['Structure normal velocity','Ion normal velocity',
                 'Electron normal velocity']
j_labels=['MMS GSM curlometer current density vs. time','Time', 
          r'Jy GSM (microA/$m^{2}$)'  ]
j_legend=['Jx','Jy','Jz']
n_labels=['MMS spatially averaged density vs. time','Time',
          'density ($cm^{-3}$)']
n_legend=['ion density','electron density']
curvature_labels=['Magnetic field curvature vs. time','Time',
                  'Curvature ($km^{-1}$)']
curvature_legend=['x component','y component','z component','total']
e_labels=['Barycenter Electric field','Time','E (mV/m)']
e_legend=['Ex','Ey','Ez']
je_labels=[r'$J\cdot E$ of structure at barycenter','Time',
           r'$J\cdot E$ ($nW/m^3$)']
je_legend=[r'$J\cdot E_{\parallel}$',r'$J\cdot E_{\perp}$',r'$J\cdot E$']

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
    edp_list=md.filenames_get(os.path.join(path,data_dir,M,edp_names_file))
    #initialize data dictionary values
    b_field[M]=np.transpose(np.array([[],[],[]])) #for all B field data
    rad[M]=np.transpose(np.array([[],[],[]])) 
    TT_time_b[M]=np.array([])
    ni[M]=np.array([])
    vi[M]=np.transpose(np.array([[],[],[]])) #for all vi data 
    presi[M]=np.array([])
    errflags_i[M]=np.array([]).astype(int)
    ne[M]=np.array([])
    ve[M]=np.transpose(np.array([[],[],[]])) #for all ve data 
    prese[M]=np.array([])
    errflags_e[M]=np.array([]).astype(int)
    e_field[M]=np.transpose(np.array([[],[],[]])) #for all E field data    
    
    for b_stub,dis_stub,des_stub,edp_stub in zip(b_list,dis_list,des_list,
                                                 edp_list):
        #create full paths
        b_file=os.path.join(path,data_dir,M,b_stub)     
        dis_file=os.path.join(path,data_dir,M,dis_stub)
        des_file=os.path.join(path,data_dir,M,des_stub)
        edp_file=os.path.join(path,data_dir,M,edp_stub)
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
        TT_time_ni_tmp,ni_tmp,temp,prestensor,dis_errs=md.get_cdf_var(dis_file,
                                       ['Epoch',
                                        'mms'+M+'_dis_numberdensity_brst',
                                        'mms'+M+'_dis_bulkv_gse_brst',
                                        'mms'+M+'_dis_prestensor_gse_brst',
                                        'mms'+M+'_dis_errorflags_brst'])
        dis_errs_interp=interp.interp1d(TT_time_ni_tmp,dis_errs,kind='nearest',
                                        fill_value="extrapolate")
        dis_errs_btime=dis_errs_interp(TT_time_tmp).astype(int)
        errflags_i[M]=np.concatenate((errflags_i[M],dis_errs_btime))
        prestmp=np.zeros_like(TT_time_ni_tmp)
        for j in range(3):
            prestmp=prestmp+prestensor[:,j,j]/3
        tmp_presi_spline=interp.CubicSpline(TT_time_ni_tmp,prestmp) #interpolate electron density data
        presi_btime_tmp=tmp_presi_spline(TT_time_tmp) #interpolate ion pressure to b-field timestamps
        presi[M]=np.concatenate((presi[M],presi_btime_tmp))
        tmp_ni_spline=interp.CubicSpline(TT_time_ni_tmp,ni_tmp) #interpolate ion density data
        ni_btime_tmp=tmp_ni_spline(TT_time_tmp) #interpolate ion density to b-field timestamps
        ni[M]=np.concatenate((ni[M],ni_btime_tmp))
        vi_tmp=temp.reshape(temp.size//3,3)
        tmp_vi_spline=interp.CubicSpline(TT_time_ni_tmp,vi_tmp) #interpolate ion veloc data
        vi_btime_tmp=tmp_vi_spline(TT_time_tmp) #interpolate ion veloc to b-field timestamps
        vi[M]=np.concatenate((vi[M],vi_btime_tmp),axis=0) 
        #read and process the ne data
        TT_time_ne_tmp,ne_tmp,prestensor,des_errs=md.get_cdf_var(des_file,
                                                                 ['Epoch',
                                            'mms'+M+'_des_numberdensity_brst',
                                            'mms'+M+'_des_prestensor_gse_brst',
                                            'mms'+M+'_des_errorflags_brst'])
        des_errs_interp=interp.interp1d(TT_time_ne_tmp,des_errs,kind='nearest',
                                        fill_value='extrapolate')
        des_errs_btime=des_errs_interp(TT_time_tmp).astype(int)
        errflags_e[M]=np.concatenate((errflags_e[M],des_errs_btime))
        prestmp=np.zeros_like(TT_time_ne_tmp)
        for j in range(3):
            prestmp=prestmp+prestensor[:,j,j]/3
        tmp_prese_interp=interp.interp1d(TT_time_ne_tmp,prestmp,
                                         fill_value="extrapolate") #interpolate electron  pressure data
        prese_btime_tmp=tmp_prese_interp(TT_time_tmp) #interpolate electron pressure to b-field timestamps
        prese[M]=np.concatenate((prese[M],prese_btime_tmp))
        tmp_ne_spline=interp.CubicSpline(TT_time_ne_tmp,ne_tmp) #interpolate electron density data
        ne_btime_tmp=tmp_ne_spline(TT_time_tmp) #interpolate electron density to b-field timestamps
        ne[M]=np.concatenate((ne[M],ne_btime_tmp))
        #read and process the E-field data
        TT_time_edp_tmp,e_field_tmp=md.get_cdf_var(edp_file,
                                               ['mms'+M+'_edp_epoch_brst_l2',
                                               'mms'+M+'_edp_dce_gse_brst_l2'])
        tmp_efield_interp=interp.interp1d(TT_time_edp_tmp,e_field_tmp,axis=0,
                                          bounds_error=False, fill_value=0.) #interpolate electric field data
        efield_btime_tmp=tmp_efield_interp(TT_time_tmp) #interpolate e-field to b-field timestamps
        e_field[M]=np.concatenate((e_field[M],efield_btime_tmp))
    #populate other necessary data dictionaries
    time_reg_b[M]=np.array(mt.TTtime2datetime(TT_time_b[M])) #time as datetime obj np arr
    ne_nozero[M]=np.where(ne[M]>ne_fudge_factor,ne[M],ne_fudge_factor) #avoid zero densities
    #transform velocities and electric fields to GSM coordinates from GSE
    vi[M]=mt.coord_transform(vi[M],'GSE','GSM',time_reg_b[M])
    e_field[M]=mt.coord_transform(e_field[M],'GSE','GSM',time_reg_b[M])    
    #roughly calculate electron velocity from curlometer
    ve_tmp=pp.electron_veloc(j_curl,TT_time_j,vi[M],ni[M],TT_time_b[M],
                            ne_nozero[M],TT_time_b[M])
    tmp_ve_spline=interp.CubicSpline(TT_time_j,ve_tmp) #interpolate electron veloc data
    ve[M]=tmp_ve_spline(TT_time_b[M]) #interpolate electron veloc to b-field timestamps    
    
    #populate j dot E total-related variables
    time_tot+=len(time_reg_b[M])*cadence/len(MMS) #at B cadence
    tmp,j_d_E_para,j_d_E_perp=pp.j_dot_e(j_curl,e_field[M],b_field[M])
    j_dot_E_para_sum['pos']+=sum(j_d_E_para[j_d_E_para>0])/len(MMS)
    j_dot_E_para_sum['neg']+=sum(j_d_E_para[j_d_E_para<0])/len(MMS)
    j_dot_E_perp_sum['pos']+=sum(j_d_E_perp[j_d_E_perp>0])/len(MMS)
    j_dot_E_perp_sum['neg']+=sum(j_d_E_perp[j_d_E_perp<0])/len(MMS)

#find potential structure candidates from MMS 1 (data smoothed a small amount)
bz_M1=ma.smoothing(b_field[MMS[0]][:,2],fixed=True,fixedwidth=6)
crossing_indices_M1=ms.find_crossings(bz_M1,time_reg_b[MMS[0]],data_gap_time)
crossing_signs_M1=ms.find_crossing_signs(bz_M1,crossing_indices_M1)
crossing_indices_M1,crossing_signs_M1=ms.refine_crossings(bz_M1,
                                                        crossing_indices_M1,
                                                        crossing_signs_M1,
                                                        extrema_width,
                                                        min_crossing_height)
crossing_structs,crossing_struct_times=ms.structure_extent(bz_M1,
                                crossing_indices_M1,time_reg_b[MMS[0]],
                                crossing_signs_M1,extrema_width,
                                min_crossing_height,len(bz_M1))
crossing_cuts,cut_struct_idxs=ms.section_maker(crossing_structs,window_padding,
                                               len(bz_M1))
#prep structure-specific dictionaries:
MMS_structure_counts['']={type_dict[0]: 0,type_dict[1]: 0,
                             type_dict[2]: 0,
                             type_dict[3]: 0,
                             type_dict[4]: 0}  
#investigate each potential structure:
for i in range(len(crossing_indices_M1)):
    if (i==16 and DEBUG): #debug option
        #check how long the code took to run
        end=time.time()
        print("Code executed in "+str(dt.timedelta(seconds=end-start)))   
        sys.exit("done with test cases")   
        
    #determine structure timings for M1
    time_struct_b=time_reg_b[MMS[0]][crossing_structs[i][0]: \
                                              crossing_structs[i][1]]
    #determine window timings for M1
    time_cut_b=time_reg_b[MMS[0]][crossing_cuts[i][0]: \
                                              crossing_cuts[i][1]]
    
    #sync all B-field data to MMS1 cadence and use a hamming window smooth
        #for the b-field data
    b_field_cut_sync={}
    b_field_struct_sync={}
    b_struct_bary=np.zeros((len(time_struct_b),3)) #b at barycenter
    rad_cut_sync={}
    rad_struct_sync={}
    vi_cut_sync={}
    vi_struct_sync={}
    vi_struct_bary=np.zeros((len(time_struct_b),3)) #vi at barycenter    
    ve_cut_sync={}
    ve_struct_sync={}
    ve_struct_bary=np.zeros((len(time_struct_b),3)) #ve at barycenter
    ni_cut_sync={}
    ni_struct_sync={}
    ni_struct_bary=np.zeros(len(time_struct_b))    
    ne_cut_sync={}
    ne_struct_sync={}
    ne_struct_bary=np.zeros(len(time_struct_b))
    presi_cut_sync={}
    presi_struct_sync={}
    presi_struct_bary=np.zeros(len(time_struct_b))  
    prese_cut_sync={}
    prese_struct_sync={}
    prese_struct_bary=np.zeros(len(time_struct_b))
    errflags_i_cut_sync={} #dis errorflags
    errflags_i_struct_sync={} #dis errorflags  
    errflags_e_cut_sync={} #des errorflags    
    errflags_e_struct_sync={} #des errorflags 
    e_field_cut_sync={}
    e_field_struct_sync={}
    e_struct_bary=np.zeros((len(time_struct_b),3)) #b at barycenter

    FPI_e_bad=np.zeros(4,dtype=bool)
    FPI_i_bad=np.zeros(4,dtype=bool)

    for j,M in enumerate(MMS):
        tmp=msc.bartlett_interp(b_field[M],time_reg_b[M],
                                                   time_cut_b)
        b_field_cut_sync[M]=ma.smoothing(tmp)
        b_field_struct_sync[M]=b_field_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1],:]
        b_struct_bary=b_struct_bary+b_field_struct_sync[M]/len(MMS)
        rad_cut_sync[M]=msc.bartlett_interp(rad[M],time_reg_b[M],
                                               time_cut_b)
        rad_struct_sync[M]=rad_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1],:]

        vi_cut_sync[M]=msc.bartlett_interp(vi[M],time_reg_b[M],
                                               time_cut_b)
        vi_struct_sync[M]=vi_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1],:]
        ve_cut_sync[M]=msc.bartlett_interp(ve[M],time_reg_b[M],
                                               time_cut_b)
        ve_struct_sync[M]=ve_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1],:]
        ni_cut_sync[M]=msc.bartlett_interp(ni[M],time_reg_b[M],
                                               time_cut_b)
        ni_struct_sync[M]=ni_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1]]
        ne_cut_sync[M]=msc.bartlett_interp(ne_nozero[M],time_reg_b[M],
                                               time_cut_b)
        ne_struct_sync[M]=ne_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1]]
        presi_cut_sync[M]=msc.bartlett_interp(presi[M],time_reg_b[M],
                                               time_cut_b)
        presi_struct_sync[M]=presi_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1]]
        prese_cut_sync[M]=msc.bartlett_interp(prese[M],time_reg_b[M],
                                               time_cut_b)
        prese_struct_sync[M]=prese_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1]]
        errflags_i_interp=interp.interp1d(mt.datetime2TTtime(time_reg_b[M]),errflags_i[M],
                                               kind="nearest")
        errflags_i_cut_sync[M]=errflags_i_interp(mt.datetime2TTtime(time_cut_b)).astype(int)
        errflags_i_struct_sync[M]=errflags_i_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1]] 
        errflags_e_interp=interp.interp1d(mt.datetime2TTtime(time_reg_b[M]),
                                          errflags_e[M],kind="nearest")
        errflags_e_cut_sync[M]=errflags_e_interp(mt.datetime2TTtime(time_cut_b)).astype(int)
        errflags_e_struct_sync[M]=errflags_e_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1]] 
        e_field_cut_sync[M]=msc.bartlett_interp(e_field[M],time_reg_b[M],
                                               time_cut_b)
        e_field_struct_sync[M]=e_field_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1],:]
        e_struct_bary=e_struct_bary+e_field_struct_sync[M]/len(MMS)
        
        #check if FPI data is struggling
        if(np.any(ne_struct_sync[M] < 0.01) or np.any(prese_struct_sync[M] <0)):
            FPI_e_bad[j]=True
        if(np.any(ni_struct_sync[M] < 0.01) or np.any(presi_struct_sync[M] <0)):
            FPI_i_bad[j]=True
            
    #compute FPI barycenter values, ignoring bad data
    total_FPI_e_sats=sum(~FPI_e_bad)
    total_FPI_i_sats=sum(~FPI_i_bad)
    for j,M in enumerate(MMS):
        if not FPI_e_bad[j]:
            ne_struct_bary=ne_struct_bary+ \
                    ne_struct_sync[M].reshape(len(ne_struct_sync[M]))/total_FPI_e_sats
            prese_struct_bary=prese_struct_bary+ \
                    prese_struct_sync[M].reshape(len(prese_struct_sync[M]))/total_FPI_e_sats   
            ve_struct_bary=ve_struct_bary+ve_struct_sync[M]/total_FPI_e_sats
        if not FPI_i_bad[j]:
            ni_struct_bary=ni_struct_bary+ \
                    ni_struct_sync[M].reshape(len(ni_struct_sync[M]))/total_FPI_i_sats
            presi_struct_bary=presi_struct_bary+ \
                    presi_struct_sync[M].reshape(len(presi_struct_sync[M]))/total_FPI_i_sats
            vi_struct_bary=vi_struct_bary+vi_struct_sync[M]/total_FPI_i_sats
    #case of no satellites having good FPI data:
    if total_FPI_e_sats == 0:
        ne_struct_bary = prese_struct_bary = np.full_like(ne_struct_bary,
                                                          np.nan,
                                                          dtype=np.double)
        ve_struct_bary = np.full_like(ve_struct_bary,np.nan,dtype=np.double)
    if total_FPI_i_sats == 0:
        ni_struct_bary = presi_struct_bary = np.full_like(ni_struct_bary,
                                                          np.nan,
                                                          dtype=np.double) 
        vi_struct_bary = np.full_like(vi_struct_bary,np.nan,dtype=np.double)
        
    #sync curlometer data also
    j_cut_sync=msc.bartlett_interp(j_curl,time_reg_jcurl,time_cut_b)
    j_struct_sync=j_cut_sync[cut_struct_idxs[i][0]:cut_struct_idxs[i][1],:]
    
    #determine structures for multispacecraft techniques
    bad_struct,crossing_time=msc.structure_crossing(b_field_struct_sync,
                                                  time_struct_b,data_gap_time)
    if(bad_struct): #not able to do multispacecraft techniques
        continue

    #do MDD analysis
    all_eigenvals,all_eigenvecs,avg_eigenvecs,std_eigenvecs=msc.MDD(b_field_struct_sync,
                                                      rad_struct_sync)
    dims_struct,multidiml,D_struct,junk=msc.structure_diml(all_eigenvals)

    #calculate ion and electron velocities normal to structure (not in invariant dir.) 
    #and check dimensionality
    print(i)
    type_str='undefined'
    vi_norm=np.zeros(len(time_struct_b))
    ve_norm=np.zeros(len(time_struct_b))
    avg_core_field=0
    b_mdd=np.empty_like(b_struct_bary) #per each eigenvector in timeseries
    b_mdd_avg=np.empty_like(b_struct_bary) #uses average eigenvectors
    if(dims_struct[2]):
        type_str="3D structure"
        for n in range(len(time_struct_b)):
            vi_norm[n]=np.linalg.norm(vi_struct_bary[n,:])
            ve_norm[n]=np.linalg.norm(ve_struct_bary[n,:])
            b_mdd[n,:]=np.transpose(all_eigenvecs[n,:,:]) @ b_struct_bary[n,:]
            b_mdd_avg[n,:]=np.transpose(avg_eigenvecs) @ b_struct_bary[n,:]
            avg_core_field=avg_core_field+b_mdd[n,2]/len(time_struct_b)
    elif(dims_struct[1]):
        type_str="2D structure"
        for n in range(len(time_struct_b)):
            vi_mdd=np.transpose(all_eigenvecs[n,:,:]) @ vi_struct_bary[n,:] #USING TIME-DEPENDENT MDD DIRECTIONS
            vi_norm[n]=np.sqrt(vi_mdd[0]*vi_mdd[0]+vi_mdd[1]*vi_mdd[1])
            ve_mdd=np.transpose(all_eigenvecs[n,:,:]) @ ve_struct_bary[n,:]
            ve_norm[n]=np.sqrt(ve_mdd[0]*ve_mdd[0]+ve_mdd[1]*ve_mdd[1])
#            vi_mdd=np.transpose(avg_eigenvecs) @ vi_struct_bary[n,:] #USING AVERAGE MDD DIRECTIONS
#            vi_norm[n]=np.sqrt(vi_mdd[0]*vi_mdd[0]+vi_mdd[1]*vi_mdd[1])
#            ve_mdd=np.transpose(avg_eigenvecs) @ ve_struct_bary[n,:]
#            ve_norm[n]=np.sqrt(ve_mdd[0]*ve_mdd[0]+ve_mdd[1]*ve_mdd[1])            
            b_mdd[n,:]=np.transpose(all_eigenvecs[n,:,:]) @ b_struct_bary[n,:]
            b_mdd_avg[n,:]=np.transpose(avg_eigenvecs) @ b_struct_bary[n,:]
            avg_core_field=avg_core_field+b_mdd[n,2]/len(time_struct_b)
    elif(dims_struct[0]):
        type_str="1D structure"
        for n in range(len(time_struct_b)):
            vi_mdd=np.transpose(all_eigenvecs[n,:,:]) @ vi_struct_bary[n,:]
            vi_norm[n]=abs(vi_mdd[0])
            ve_mdd=np.transpose(all_eigenvecs[n,:,:]) @ ve_struct_bary[n,:]
            ve_norm[n]=abs(ve_mdd[0])
#            vi_mdd=np.transpose(avg_eigenvecs) @ vi_struct_bary[n,:]
#            vi_norm[n]=abs(vi_mdd[0])
#            ve_mdd=np.transpose(avg_eigenvecs) @ ve_struct_bary[n,:]
#            ve_norm[n]=abs(ve_mdd[0])
            b_mdd[n,:]=np.transpose(all_eigenvecs[n,:,:]) @ b_struct_bary[n,:]
            b_mdd_avg[n,:]=np.transpose(avg_eigenvecs) @ b_struct_bary[n,:]
            avg_core_field=avg_core_field+b_mdd[n,2]/len(time_struct_b)
    if(multidiml):
        type_str=type_str+" with some smaller-dimensional qualities"
    str_avg_core_field=f"{avg_core_field:.2f}"

    #do STD analysis
    velocs,optimal=msc.STD(b_field_cut_sync,time_cut_b,cut_struct_idxs[i],
                               rad_struct_sync,all_eigenvals,all_eigenvecs,
                               dims_struct,min_crossing_height)
    #do STD analysis with average MDD results
#    avg_eigenvals=np.average(all_eigenvals,axis=0)
#    velocs,optimal=msc.STD_avged(b_field_cut_sync,time_cut_b,cut_struct_idxs[i],
#                               rad_struct_sync,avg_eigenvals,avg_eigenvecs,
#                               dims_struct,min_crossing_height)
    vtot=np.linalg.norm(velocs,axis=1)
    avg_vtot=np.average(vtot) 
    sgned_avg_vtot=np.sign(np.average(velocs[:,0]))*avg_vtot #positive if average vx positive
    
    #calculate other properties of the structure (kind, size, etc.)
    #determine error-flags for des and dis data
    err_i_label="Ion moments error flags:\n"   
    for M in MMS:
        bool_errs=np.zeros(32,dtype=bool)
        err_i_label+="MMS"+M+"- "
        for n in range(len(errflags_i_struct_sync[M])):
            err_b_str="{0:b}".format(errflags_i_struct_sync[M][n])
            for j in range(len(err_b_str)):
                if (err_b_str[len(err_b_str)-1-j] is '1'):
                    bool_errs[j]=True
        for j in range(len(bool_errs)):
            if bool_errs[j]:
                err_i_label+=str(j)+' '
    err_i_label+='\n'
    err_e_label="Electron moments error flags:\n"   
    for M in MMS:
        bool_errs=np.zeros(32,dtype=bool)
        err_e_label+="MMS"+M+"- "
        for n in range(len(errflags_e_struct_sync[M])):
            err_b_str="{0:b}".format(errflags_e_struct_sync[M][n])
            for j in range(len(err_b_str)):
                if (err_b_str[len(err_b_str)-1-j] is '1'):
                    bool_errs[j]=True
        for j in range(len(bool_errs)):
            if bool_errs[j]:
                err_e_label+=str(j)+' '
    err_e_label+='\n'
                
            
    #calculate plasma beta
    betai_struct_bary=pp.beta(presi_struct_bary,b_struct_bary)
    betai_avg=np.average(betai_struct_bary)
    str_beta_avg=f"{betai_avg:.2f}"  #string formatting
    #calculate the curvature of the magnetic field
    curvature_struct=msc.curvature(b_field_struct_sync,rad_struct_sync,
                                   b_struct_bary)
    total_curvature_struct=np.linalg.norm(curvature_struct,axis=1)
    #calculate JdotE, perp and para, and add to total
    jE_struct,jE_para_struct,jE_perp_struct=pp.j_dot_e(j_struct_sync,
                                                       e_struct_bary,
                                                       b_struct_bary)
    jE_para_avg=np.average(jE_para_struct)
    jE_perp_avg=np.average(jE_perp_struct)
    time_structs+=len(time_struct_b)*cadence
    j_dot_E_struct_para_sum['pos']+=sum(jE_para_struct[jE_para_struct>0])
    j_dot_E_struct_para_sum['neg']+=sum(jE_para_struct[jE_para_struct<0])
    j_dot_E_struct_perp_sum['pos']+=sum(jE_perp_struct[jE_perp_struct>0])
    j_dot_E_struct_perp_sum['neg']+=sum(jE_perp_struct[jE_perp_struct<0])

    #calculate approximate guide field
    gf,tmp=msc.guide_field(b_field_struct_sync)
    str_gf=f"{gf:.2f}"
    #calculate approximate electron and ion plasma frequencies and skin depths
    we_struct_bary=pp.plasma_frequency(ne_struct_bary,const.m_e)
    wp_struct_bary=pp.plasma_frequency(ni_struct_bary,const.m_p) #assuming all ions are protons (valid?)
    de_struct_bary=pp.inertial_length(we_struct_bary)
    dp_struct_bary=pp.inertial_length(wp_struct_bary) 
    de_avg=np.average(de_struct_bary)
    dp_avg=np.average(dp_struct_bary)
    str_de_avg=f"{de_avg:.1f}"  #string formatting
    str_dp_avg=f"{dp_avg:.1f}"  #string formatting
    #determine signs of vex and jy
    jy_sign,jy_qual=ma.find_avg_signs(j_struct_sync[:,1])
    v_sign,v_qual=ma.find_avg_signs(velocs[:,0]) #x component of structure normal
    #determine type of structure, size of structure
    crossing_type,type_flag=ms.structure_classification(crossing_signs_M1[i],
                                                        v_sign,v_qual,jy_sign,
                                                        jy_qual,quality_min)
        
    crossing_duration=(time_struct_b[-1]-time_struct_b[0]).total_seconds()
    crossing_size=ms.structure_sizer([time_struct_b[0],
                                          time_struct_b[-1]],vtot)
    str_crossing_size=f"{crossing_size:.1f}"  #string formatting
    #normalize size and (normal) velocity by ion and electron scales
    crossing_size_de=crossing_size/de_avg
    crossing_size_dp=crossing_size/dp_avg
    avg_vtot_e=np.average(vtot/ve_norm)
    avg_vtot_p=np.average(vtot/vi_norm)
    
    #add the structure to the total counts
    MMS_structure_counts[''][type_dict[type_flag]] += 1
    MMS_structures.append(Structure(type_dict[type_flag],crossing_duration,
                                                crossing_size,crossing_size_de,
                                                crossing_size_dp,
                                                avg_core_field,gf,avg_vtot,
                                                sgned_avg_vtot, avg_vtot_e,
                                                avg_vtot_p,type_str,
                                                jE_para_avg,jE_perp_avg))
    #determine the j dot E breakdown for each 'real' structure
    if type_flag == 0:
        time_plasmoid+=len(time_struct_b)*cadence
        j_dot_E_plasmoid_para['pos']+=sum(jE_para_struct[jE_para_struct>0])
        j_dot_E_plasmoid_para['neg']+=sum(jE_para_struct[jE_para_struct<0])
        j_dot_E_plasmoid_perp['pos']+=sum(jE_perp_struct[jE_perp_struct>0])
        j_dot_E_plasmoid_perp['neg']+=sum(jE_perp_struct[jE_perp_struct<0])
    if type_flag == 1:
        time_pullcs+=len(time_struct_b)*cadence
        j_dot_E_pullcs_para['pos']+=sum(jE_para_struct[jE_para_struct>0])
        j_dot_E_pullcs_para['neg']+=sum(jE_para_struct[jE_para_struct<0])
        j_dot_E_pullcs_perp['pos']+=sum(jE_perp_struct[jE_perp_struct>0])
        j_dot_E_pullcs_perp['neg']+=sum(jE_perp_struct[jE_perp_struct<0])    
    if type_flag == 2:
        time_pushcs+=len(time_struct_b)*cadence
        j_dot_E_pushcs_para['pos']+=sum(jE_para_struct[jE_para_struct>0])
        j_dot_E_pushcs_para['neg']+=sum(jE_para_struct[jE_para_struct<0])
        j_dot_E_pushcs_perp['pos']+=sum(jE_perp_struct[jE_perp_struct>0])
        j_dot_E_pushcs_perp['neg']+=sum(jE_perp_struct[jE_perp_struct<0])       
        
    if(REPLOT):
        #structure information for plot
        jy_sign_label="jy sign is "+str(jy_sign)+" with quality "+ \
            str(jy_qual)+"\n"
        v_sign_label="vx sign is "+str(v_sign)+" with quality " \
                        +str(v_qual)+"\n"
        crossing_sign_label="Crossing type: "+crossing_type+"\n"
        crossing_size_label="Crossing size: "+str_crossing_size+" km"+"\n"
        crossing_de_label="Average electron inertial length: "+ \
                            str_de_avg+" km"+"\n"
        crossing_dp_label="Average proton inertial length: "+ \
                            str_dp_avg+" km"+"\n" 
        crossing_beta_label="Average ion beta: "+str_beta_avg+"\n"
        core_field_label="Average core field: "+str_avg_core_field+" nT\n"
        guide_field_label="Guide field: "+str_gf+" nT\n"
        #plot it 
        mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
        plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
        gridsize=(7,2)
        fig=plt.figure(figsize=(16,16)) #width,height
        ax1=plt.subplot2grid(gridsize,(0,0))
        ax2=plt.subplot2grid(gridsize,(1,0))   
        ax3=plt.subplot2grid(gridsize,(2,0)) 
        ax4=plt.subplot2grid(gridsize,(3,0))
        ax5=plt.subplot2grid(gridsize,(4,0))
        ax67=plt.subplot2grid(gridsize,(5,0),rowspan=2)
        ax67.axis('off')
        ax8=plt.subplot2grid(gridsize,(0,1))
        ax9=plt.subplot2grid(gridsize,(1,1))
        ax10=plt.subplot2grid(gridsize,(2,1))
        ax11=plt.subplot2grid(gridsize,(3,1))
        ax12=plt.subplot2grid(gridsize,(4,1))
        ax13=plt.subplot2grid(gridsize,(5,1))
        ax14=plt.subplot2grid(gridsize,(6,1))
        #plot joined and smoothed B-fields
        for M in MMS:  
            bz=b_field_cut_sync[M][:,2]
            mmsp.tseries_plotter(fig,ax1,time_cut_b,bz,
                                 labels=b_label,
                                 lims=[min(time_cut_b),
                                       max(time_cut_b)],
                                 legend=M) 
        #plot the eigenvalues
        for j in range(len(all_eigenvals[0,:])):
            eigenvals=all_eigenvals[:,j]
            mmsp.tseries_plotter(fig,ax2,time_struct_b,eigenvals,
                                 labels=eigenval_label,
                                 lims=[min(time_cut_b),
                                       max(time_cut_b)],
                                 legend=eigenval_legend[j],logscale=True)  
            
        #plot the eigenvectors
        ax_subset=[ax3,ax4,ax5]
        for j in range(3):
            for k in range(3):
                mmsp.tseries_plotter(fig,ax_subset[j],time_struct_b,
                                     all_eigenvecs[:,k,j],
                                     labels=eigenvec_label[j],
                                     lims=[min(time_cut_b),max(time_cut_b)],
                                     legend=eigenvec_legend[k])
        for j in range(3):
            mmsp.tseries_plotter(fig,ax8,time_cut_b,j_cut_sync[:,j],j_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=j_legend[j]) #plot j
        #plot the structure velocities
        for j in range(3):
            mmsp.tseries_plotter(fig,ax9,time_struct_b,
                                 velocs[:,j],
                                 labels=veloc_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=veloc_legend[j])
        mmsp.tseries_plotter(fig,ax9,time_struct_b,
                                 vtot,
                                 labels=veloc_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=veloc_legend[3]) 
        #plot comparison of total normal velocities to ion/electron normal velocities
        mmsp.tseries_plotter(fig,ax10,time_struct_b,
                                 vtot,
                                 labels=vcompare_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=vcompare_legend[0])
        mmsp.tseries_plotter(fig,ax10,time_struct_b,
                                 vi_norm,
                                 labels=vcompare_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=vcompare_legend[1])   
        mmsp.tseries_plotter(fig,ax10,time_struct_b,
                                 ve_norm,
                                 labels=vcompare_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=vcompare_legend[2])
        #plot densities
        mmsp.tseries_plotter(fig,ax11,time_struct_b,
                                 ni_struct_bary,
                                 labels=n_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=n_legend[0])  
        mmsp.tseries_plotter(fig,ax11,time_struct_b,
                                 ne_struct_bary,
                                 labels=n_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=n_legend[1])  
        #plot curvature components
        for j in range(3):
            mmsp.tseries_plotter(fig,ax12,time_struct_b,
                                     curvature_struct[:,j],
                                     labels=curvature_labels,
                                     lims=[min(time_cut_b),max(time_cut_b)],
                                     legend=curvature_legend[j]) 
        #plot total curvature 
        mmsp.tseries_plotter(fig,ax12,time_struct_b,
                                 total_curvature_struct,
                                 labels=curvature_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=curvature_legend[3])  
        #plot electric field
        for j in range(3):
            mmsp.tseries_plotter(fig,ax13,time_struct_b,
                                     e_struct_bary[:,j],
                                     labels=e_labels,
                                     lims=[min(time_cut_b),max(time_cut_b)],
                                     legend=e_legend[j]) 
        #plot J dot E components
        mmsp.tseries_plotter(fig,ax14,time_struct_b,
                                 jE_para_struct,
                                 labels=je_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=je_legend[0])  
        mmsp.tseries_plotter(fig,ax14,time_struct_b,
                                 jE_perp_struct,
                                 labels=je_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=je_legend[1])
        mmsp.tseries_plotter(fig,ax14,time_struct_b,
                                 jE_struct,
                                 labels=je_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=je_legend[2])           
        #add horizontal and vertical lines to plot (crossing + extent)
        mmsp.line_maker([ax1,ax2,ax3,ax4,ax5,ax8,ax9,ax10,ax11,ax12,ax13,ax14],
                        time=crossing_time,edges=crossing_struct_times[i],
                        horiz=0.)
         #add categorization information to plot
        ax67.text(0.5,0.5,type_str+'\n'+jy_sign_label+v_sign_label+ \
                 crossing_sign_label+crossing_size_label+crossing_dp_label+ \
                 crossing_de_label+crossing_beta_label+core_field_label+\
                 guide_field_label+err_i_label+err_e_label,wrap=True,
                 transform=ax67.transAxes,fontsize=16,ha='center',va='center')       
        fig.savefig(os.path.join(timeseries_out_directory,'MMS'+'_'+ \
                                plot_out_name+str(i)+".png"), 
                                bbox_inches='tight')
        plt.close(fig="all")    

    if(i==6 or i==35 or i==80): #example plots for paper
        gridsize=(3,1)
        fig=plt.figure(figsize=(5,4)) #width,height
        ax1=plt.subplot2grid(gridsize,(0,0))
        ax2=plt.subplot2grid(gridsize,(1,0))   
        ax3=plt.subplot2grid(gridsize,(2,0)) 
        start_time=time_cut_b[0]
        time_cut_delta=time_cut_b - start_time
        time_cut_s=time_cut_delta/dt.timedelta(microseconds=1)/1e6
        crossing_time_s=(crossing_time-start_time)/dt.timedelta(microseconds=1)/1e6
        crossing_struct_times_s=[(crossing_struct_times[i][0]-start_time)/dt.timedelta(microseconds=1)/1e6,
                                  (crossing_struct_times[i][1]-start_time)/dt.timedelta(microseconds=1)/1e6]
        for M in MMS: #plot smoothed B-field  
            bz=b_field_cut_sync[M][:,2]
            mmsp.tseries_plotter(fig,ax1,time_cut_b,bz,
                                 labels=['','',b_label[2]],
                                 lims=[min(time_cut_b),
                                       max(time_cut_b)],
                                 legend=M)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax1.tick_params(labelbottom=False)
        for j in range(3): #plot the structure velocities
            mmsp.tseries_plotter(fig,ax2,time_struct_b,
                                 velocs[:,j],
                                 labels=['','',r'$v_{normal}$ (km/s)'],
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=veloc_legend[j])
        mmsp.tseries_plotter(fig,ax2,time_struct_b,
                                 vtot,
                                 labels=['','',r'$v_{normal}$ (km/s)'],
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=veloc_legend[3])  
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax2.tick_params(labelbottom=False)
        for j in range(3):  #plot j  
            mmsp.basic_plotter(ax3,time_cut_s,j_cut_sync[:,j],labels=['',r'$s$ after {}'.format(start_time.strftime("%b %d %Y %H:%M:%S.%f")),
                                 r'$J_y$ GSM $(\mu A/m^2)$ '],
                                 xlims=[min(time_cut_s),max(time_cut_s)],
                                 legend=j_legend[j]) 
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#        ax3.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0e'))
        #add horizontal and vertical lines to plot (crossing + extent)
        mmsp.line_maker([ax1,ax2],time=crossing_time,edges=crossing_struct_times[i],
                        horiz=0.)
        mmsp.line_maker([ax3],time=crossing_time_s,edges=crossing_struct_times_s,
                        horiz=0.)
        fig.savefig(os.path.join(timeseries_out_directory,'MMS'+'_'+ \
                                'paperfig'+str(i)+".svg"), 
                                bbox_inches='tight')
        plt.close(fig="all") 
        

""" STATISTICAL PART """

#make bar chart of the different types of structures
fig_bar,ax_bar=plt.subplots(figsize=(5,4))
mmsp.bar_charter(ax_bar,MMS_structure_counts,['Types of structures seen by MMS',
                                         '',
                                         'Counts']) 
fig_bar.savefig(os.path.join(statistics_out_directory,
                             "types_bar_chart"+".svg"),bbox_inches='tight')
plt.close(fig='all')

if(REHIST):
    #make histograms of the x-lengths of all structures
    structure_kinds=type_dict.values()
    mmsp.msc_structure_hist_maker(MMS_structures,"size",hists_out_directory,nbins,
                              structure_kinds)    
    mmsp.msc_structure_hist_maker(MMS_structures,"size",hists_out_directory,
                              nbins_small,structure_kinds, log=True)
    #make histograms of the ion and electron skin depths of all structures
    mmsp.msc_structure_hist_maker(MMS_structures,"ion_skin_depth",hists_out_directory,nbins,
                              structure_kinds)    
    mmsp.msc_structure_hist_maker(MMS_structures,"electron_skin_depth",hists_out_directory,
                              nbins,structure_kinds)    
    #make histograms of the time durations of all structures
    mmsp.msc_structure_hist_maker(MMS_structures,"duration",hists_out_directory,
                                  nbins,structure_kinds)   
    mmsp.msc_structure_hist_maker(MMS_structures,"duration",hists_out_directory,
                                  nbins_small,structure_kinds, log=True)
    #make histograms of the normalized x-lengths of all structures
    mmsp.msc_structure_hist_maker(MMS_structures,"electron_normalized_size",
                                  hists_out_directory,nbins,structure_kinds)  
    mmsp.msc_structure_hist_maker(MMS_structures,"ion_normalized_size",
                                  hists_out_directory,nbins,structure_kinds) 
    mmsp.msc_structure_hist_maker(MMS_structures,"electron_normalized_size",
                                  hists_out_directory,nbins_small,structure_kinds,
                                  log=True)  
    mmsp.msc_structure_hist_maker(MMS_structures,"ion_normalized_size",
                                  hists_out_directory,nbins_small,structure_kinds,
                                  log=True)
    #make histograms of the guide field strengths of all structures
    mmsp.msc_structure_hist_maker(MMS_structures,'guide_field',hists_out_directory,
                              nbins_small,structure_kinds)
    #make histograms of the core field strengths of all structures
    mmsp.msc_structure_hist_maker(MMS_structures,'core_field',hists_out_directory,
                              nbins_small,structure_kinds)      
    # make histograms of the velocities of all structures
    mmsp.msc_structure_hist_maker(MMS_structures,'normal_speed',
                                  hists_out_directory,nbins,structure_kinds) 
    mmsp.msc_structure_hist_maker(MMS_structures,'normal_speed',
                                  hists_out_directory,nbins_small,structure_kinds,
                                  log=True) 
    #make histograms of the normalized velocities of all structures
    mmsp.msc_structure_hist_maker(MMS_structures,'electron_normalized_speed',
                                  hists_out_directory,nbins,structure_kinds)
    mmsp.msc_structure_hist_maker(MMS_structures,'ion_normalized_speed',
                                  hists_out_directory,nbins,structure_kinds)
    mmsp.msc_structure_hist_maker(MMS_structures,'electron_normalized_speed',
                                  hists_out_directory,nbins_small,structure_kinds,
                                  log=True)
    mmsp.msc_structure_hist_maker(MMS_structures,'ion_normalized_speed',
                                  hists_out_directory,nbins_small,structure_kinds,
                                  log=True)
    #make histograms of the j dot E of all structures
    mmsp.msc_structure_hist_maker(MMS_structures,'j_dot_E_parallel',
                                  hists_out_directory,nbins,structure_kinds)
    mmsp.msc_structure_hist_maker(MMS_structures,'j_dot_E_perpendicular',
                                  hists_out_directory,nbins,structure_kinds)
    ##make scatter plot of guide field strength vs structure size 
    #mmsp.msc_structure_scatter_maker(MMS_structures,'size','guide_field',
    #                                 scatters_out_directory,structure_kinds)  
    ##make scatter plot of core field strength vs structure size
    #mmsp.msc_structure_scatter_maker(MMS_structures,'size','core_field',
    #                                 scatters_out_directory,structure_kinds) 
    # make scatter plot of signed normal speed vs structure size 
    mmsp.msc_structure_scatter_maker(MMS_structures,'size','signed_normal_speed',
                                     scatters_out_directory,structure_kinds)
    ##make scatter plot of normal speed vs structure duration
    #mmsp.msc_structure_scatter_maker(MMS_structures,'duration','normal_speed',
    #                                 scatters_out_directory,structure_kinds)
    ## make scatter plot of normalized speeds vs normalized structure size
    #mmsp.msc_structure_scatter_maker(MMS_structures,'electron_normalized_size',
    #                                 'electron_normalized_speed',
    #                                 scatters_out_directory,structure_kinds)
    #mmsp.msc_structure_scatter_maker(MMS_structures,'ion_normalized_size',
    #                                 'ion_normalized_speed',
    #                                 scatters_out_directory,structure_kinds)
    ##make scatter plot comparing ion-normalized and electron-normalized attributes
    #mmsp.msc_structure_scatter_maker(MMS_structures,'electron_normalized_size',
    #                                 'ion_normalized_size',
    #                                 scatters_out_directory,structure_kinds)
    #mmsp.msc_structure_scatter_maker(MMS_structures,'electron_normalized_speed',
    #                                 'ion_normalized_speed',
    #                                 scatters_out_directory,structure_kinds)
    ## make scatter plot of the j dot E vs structure size 
    #mmsp.msc_structure_scatter_maker(MMS_structures,'size','j_dot_E_parallel',
    #                                 scatters_out_directory,structure_kinds)
    #mmsp.msc_structure_scatter_maker(MMS_structures,'size','j_dot_E_perpendicular',
    #                                 scatters_out_directory,structure_kinds)
    ## make scatter plot of the j dot E vs ion-normalized structure size 
    #mmsp.msc_structure_scatter_maker(MMS_structures,'ion_normalized_size',
    #                                 'j_dot_E_parallel',scatters_out_directory,
    #                                 structure_kinds)
    #mmsp.msc_structure_scatter_maker(MMS_structures,'ion_normalized_size',
    #                                 'j_dot_E_perpendicular',
    #                                 scatters_out_directory,structure_kinds)
    ##make scatter plot of the j dot E vs core field 
    #mmsp.msc_structure_scatter_maker(MMS_structures,'core_field',
    #                                 'j_dot_E_parallel',scatters_out_directory,
    #                                 structure_kinds)
    #mmsp.msc_structure_scatter_maker(MMS_structures,'core_field',
    #                                 'j_dot_E_perpendicular',
    #                                 scatters_out_directory,structure_kinds)
    ## make scatter plot of the j dot E vs guide field 
    #mmsp.msc_structure_scatter_maker(MMS_structures,'guide_field',
    #                                 'j_dot_E_parallel',scatters_out_directory,
    #                                 structure_kinds)
    #mmsp.msc_structure_scatter_maker(MMS_structures,'guide_field',
    #                                 'j_dot_E_perpendicular',
    #                                 scatters_out_directory,structure_kinds)

#make histograms of the j dot E (for testing j dot E hist quicker)
mmsp.msc_structure_hist_maker(MMS_structures,'j_dot_E_parallel',
                              hists_out_directory,30,structure_kinds)
''' J dot E INFO '''
print(j_dot_E_para_sum)
print(j_dot_E_perp_sum)
print(j_dot_E_struct_para_sum)
print(j_dot_E_struct_perp_sum)
print(j_dot_E_plasmoid_para)
print(j_dot_E_plasmoid_perp)
print(j_dot_E_pullcs_para)
print(j_dot_E_pullcs_perp)
print(j_dot_E_pushcs_para)
print(j_dot_E_pushcs_perp)

#make pie charts of the J dot E breakdowns
fig_pie,ax_pie=plt.subplots()
cmap = plt.get_cmap("tab20c")
w_width=.3

#set data for the pie of overall (summed) J dot E breakdown
j_dot_E_para_tot=j_dot_E_para_sum['pos']+j_dot_E_para_sum['neg']
j_dot_E_perp_tot=j_dot_E_perp_sum['pos']+j_dot_E_perp_sum['neg']
j_dot_E_para_struct_tot=j_dot_E_struct_para_sum['pos']+j_dot_E_struct_para_sum['neg']
j_dot_E_para_out_tot=j_dot_E_para_tot-j_dot_E_para_struct_tot
j_dot_E_perp_struct_tot=j_dot_E_struct_perp_sum['pos']+j_dot_E_struct_perp_sum['neg']
j_dot_E_perp_out_tot=j_dot_E_perp_tot-j_dot_E_perp_struct_tot

#set colors for the overall pie
incolors=cmap([0,1,13,12])
#plot overall pie
mmsp.pie_plotter(ax_pie,[j_dot_E_para_struct_tot,j_dot_E_para_out_tot,j_dot_E_perp_out_tot,
                 j_dot_E_perp_struct_tot],incolors,
                 r"$J \cdot E$ breakdown outside and within structures",
                 wdglabels=[r"$J_{\parallel} E_{\parallel}$"+ "\n within",
                  r"$J_{\parallel} E_{\parallel}$"+ "\n outside",
                  r"$J_{\perp} \cdot E_{\perp}$"+ "\n outside",
                  r"$J_{\perp} \cdot E_{\perp}$"+ "\n within"],
                 radius=1,width=2*w_width)
fig_pie.savefig(os.path.join(statistics_out_directory,
                             "j_dot_e_pie_overall"+".svg"),bbox_inches='tight')
plt.close(fig='all')

fig_pie,ax_pie=plt.subplots()
#set data for positive-negative breakdown of j_parallel e_parallel
j_dot_E_para_out_pos=j_dot_E_para_sum['pos']-j_dot_E_struct_para_sum['pos']
j_dot_E_para_out_neg=np.abs(j_dot_E_para_sum['neg']-j_dot_E_struct_para_sum['neg'])

#set colors for j_parallel e_parallel pie
outcolors=cmap(np.array([0,3])*4)
incolors=cmap(np.array([0,3])*4+1)
angle=90
#plot j_parallel e_parallel pie
mmsp.pie_plotter(ax_pie,[j_dot_E_para_out_pos,j_dot_E_para_out_neg],outcolors,
                 r"$J_{\parallel} E_{\parallel}$ contributions outside and within structures",
                 wdglabels=["positive \n"+r"$J_{\parallel} E_{\parallel}$",
                  "negative \n"+r"$J_{\parallel} E_{\parallel}$"],
                  width=w_width,startangle=angle)
mmsp.pie_plotter(ax_pie,[j_dot_E_struct_para_sum['pos'],np.abs(j_dot_E_struct_para_sum['neg'])],incolors,
                  r"$J_{\parallel} E_{\parallel}$ contributions outside and within structures",                
                  labels=[r"positive $J_{\parallel} E_{\parallel}$"+"\n within structures",
                  r"negative $J_{\parallel} E_{\parallel}$"+"\n within structures"],
                  radius=1-w_width, width=w_width,startangle=angle)

fig_pie.savefig(os.path.join(statistics_out_directory,
                             "j_dot_e_pie_para"+".svg"),bbox_inches='tight')
plt.close(fig='all')

fig_pie,ax_pie=plt.subplots()
#set data for positive-negative breakdown of j_perp e_perp
j_dot_E_perp_out_pos=j_dot_E_perp_sum['pos']-j_dot_E_struct_perp_sum['pos']
j_dot_E_perp_out_neg=np.abs(j_dot_E_perp_sum['neg']-j_dot_E_struct_perp_sum['neg'])

#plot j_perp e_perp pie
mmsp.pie_plotter(ax_pie,[j_dot_E_perp_out_pos,j_dot_E_perp_out_neg],outcolors,
                 r"$J_{\perp} \cdot E_{\perp}$ contributions outside and within structures",
                 wdglabels=["positive \n"+r"$J_{\perp} \cdot E_{\perp}$",
                  "negative \n"+r"$J_{\perp} \cdot E_{\perp}$"],
                  width=w_width,startangle=angle)
mmsp.pie_plotter(ax_pie,[j_dot_E_struct_perp_sum['pos'],np.abs(j_dot_E_struct_perp_sum['neg'])],incolors,
                 r"$J_{\perp} \cdot E_{\perp}$ contributions outside and within structures",       
                 labels=[r"positive $J_{\perp} \cdot E_{\perp}$"+ "\n within structures",
                 r"negative $J_{\perp} \cdot E_{\perp}$"+"\n within structures"],
                 radius=1-w_width, width=w_width,startangle=angle)

fig_pie.savefig(os.path.join(statistics_out_directory,
                             "j_dot_e_pie_perp"+".svg"),bbox_inches='tight')
plt.close(fig='all')

print(MMS_structure_counts)
#check how long the code took to run
end=time.time()
print("Code executed in "+str(dt.timedelta(seconds=end-start))) 
