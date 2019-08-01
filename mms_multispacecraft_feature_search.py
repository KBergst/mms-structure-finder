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
statistics_out_directory=os.path.join(path,plot_out_directory,"statistics")
hists_out_directory=os.path.join(statistics_out_directory,"hists")
scatters_out_directory=os.path.join(statistics_out_directory,"scatters")

#parameters
data_gap_time=dt.timedelta(milliseconds=10) #amount of time to classify 
                                                  #something as a data gap
extrema_width=10 #number of points to compare on each side to declare an extrema
smoothed_extrema_width=10 #number of points to compare on each side to declare an extrema for the smoothed data
min_crossing_height=0.1 #expected nT error in region of interest as per documentation
window_padding=20 #number of indices to add to each side of window
ne_fudge_factor=0.001 #small amount of density to add to avoid NaN velocities
quality_min=0.5 #used in structure_classification, minimum accepted quality
nbins_small=10 #number of bins for log-scale histograms and other small hists
nbins=15 #number of bins for the size histograms

DEBUG=1 #chooses whether to stop at iteration 15 or not
REPLOT=1 #chooses whether to regenerate the plots or not

###### CLASS DEFINITIONS ######################################################
class Structure:
    
    #initializer
    def __init__(self,kind,size,size_e,size_p,gf,vel,vel_e,vel_p):
        self.kind=kind
        self.size=size
        self.electron_normalized_size=size_e
        self.ion_normalized_size=size_p
        self.guide_field=gf
        self.normal_speed=vel
        self.electron_normalized_speed=vel_e
        self.ion_normalized_speed=vel_p
        
    #for pluralization in outputs
    plurals={'size':'sizes',
                 'electron_normalized_size':'electron-normalized sizes',
                 'ion_normalized_size':'ion-normalized sizes',
                  'kind':'kinds',
                  'guide_field':'guide fields',
                  'normal_speed':'normal speeds',
                  'electron_normalized_speed':'electron-normalized speeds',
                  'ion_normalized_speed':'ion-normalized speeds'
                }
    
    #for getting units
    units={'size':'km',
                'electron_normalized_size':'electron inertial lengths',
                'ion_normalized_size':'ion inertial lengths',
                'kind':'',
                'guide_field':'nT',
                'normal_speed':'km/s',
                'electron_normalized_speed':'unitless',
                'ion_normalized_speed':'unitless'                
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

b_field={} #dictionary that will contain all the different magnetic field data
rad={} #dictionary holding resampled spacecraft position data (same time cadence as that satellite's b-field)
TT_time_b={} #for magnetic field timeseries in MMS times (TT2000)
time_reg_b={} #for magnetic field timeseries in datetime objects
ni={} #for ion density
vi={} #will be interpolated to b-field cadence (fit w/ spline)
ne={} #for electron density
ne_nozero={} #to remove underflows from the electron density data
ve={} #for curlometer electron velocity

j_curl=np.transpose(np.array([[],[],[]]))  #for all j data
time_reg_jcurl=np.array([])

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
n_labels=['MMS spatially averaged density vs. time','Time',
          'density ($cm^{-3}$)']
n_legend=['ion density','electron density']
curvature_labels=['Magnetic field curvature vs. time','Time',
                  'Curvature ($km^{-1}$)']
curvature_legend=['x component','y component','z component','total']

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
    #initialize data dictionary values
    b_field[M]=np.transpose(np.array([[],[],[]])) #for all B field data
    rad[M]=np.transpose(np.array([[],[],[]])) 
    TT_time_b[M]=np.array([])
    ni[M]=np.array([])
    vi[M]=np.transpose(np.array([[],[],[]])) #for all vi data 
    ne[M]=np.array([])
    ve[M]=np.transpose(np.array([[],[],[]])) #for all ve data 
    
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
        TT_time_ni_tmp,ni_tmp,temp=md.get_cdf_var(dis_file,
                                       ['Epoch',
                                        'mms'+M+'_dis_numberdensity_brst',
                                        'mms'+M+'_dis_bulkv_gse_brst'])
        tmp_ni_spline=interp.CubicSpline(TT_time_ni_tmp,ni_tmp) #interpolate ion density data
        ni_btime_tmp=tmp_ni_spline(TT_time_tmp) #interpolate ion density to b-field timestamps
        ni[M]=np.concatenate((ni[M],ni_btime_tmp))
        vi_tmp=temp.reshape(temp.size//3,3)
        tmp_vi_spline=interp.CubicSpline(TT_time_ni_tmp,vi_tmp) #interpolate ion veloc data
        vi_btime_tmp=tmp_vi_spline(TT_time_tmp) #interpolate ion veloc to b-field timestamps
        vi[M]=np.concatenate((vi[M],vi_btime_tmp),axis=0) 
        #read and process the ne data
        TT_time_ne_tmp,ne_tmp=md.get_cdf_var(des_file,['Epoch',
                                            'mms'+M+'_des_numberdensity_brst'])
        tmp_ne_spline=interp.CubicSpline(TT_time_ne_tmp,ne_tmp) #interpolate electron density data
        ne_btime_tmp=tmp_ne_spline(TT_time_tmp) #interpolate ion density to b-field timestamps
        ne[M]=np.concatenate((ne[M],ne_btime_tmp))
    
    #populate other necessary data dictionaries
    time_reg_b[M]=np.array(mt.TTtime2datetime(TT_time_b[M])) #time as datetime obj np arr
    ne_nozero[M]=np.where(ne[M]>ne_fudge_factor,ne[M],ne_fudge_factor) #avoid zero densities
    #transform velocities to GSM coordinates from GSE
    vi[M]=mt.coord_transform(vi[M],'GSE','GSM',time_reg_b[M])
    #roughly calculate electron velocity from curlometer
    ve_tmp=pp.electron_veloc(j_curl,TT_time_j,vi[M],ni[M],TT_time_b[M],
                            ne_nozero[M],TT_time_b[M])
    tmp_ve_spline=interp.CubicSpline(TT_time_j,ve_tmp) #interpolate electron veloc data
    ve[M]=tmp_ve_spline(TT_time_b[M]) #interpolate electron veloc to b-field timestamps    

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

    
    for M in MMS:
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
        vi_struct_bary=vi_struct_bary+vi_struct_sync[M]/len(MMS)
        ve_cut_sync[M]=msc.bartlett_interp(ve[M],time_reg_b[M],
                                               time_cut_b)
        ve_struct_sync[M]=ve_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1],:]
        ve_struct_bary=ve_struct_bary+ve_struct_sync[M]/len(MMS)
        ni_cut_sync[M]=msc.bartlett_interp(ni[M],time_reg_b[M],
                                               time_cut_b)
        ni_struct_sync[M]=ni_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1]]
        ni_struct_bary=ni_struct_bary+ \
                    ni_struct_sync[M].reshape(len(ni_struct_sync[M]))/len(MMS)
        ne_cut_sync[M]=msc.bartlett_interp(ne_nozero[M],time_reg_b[M],
                                               time_cut_b)
        ne_struct_sync[M]=ne_cut_sync[M][cut_struct_idxs[i][0]: \
                                              cut_struct_idxs[i][1]]
        ne_struct_bary=ne_struct_bary+ \
                    ne_struct_sync[M].reshape(len(ne_struct_sync[M]))/len(MMS)
        
    #sync curlometer data also
    j_cut_sync=msc.bartlett_interp(j_curl,time_reg_jcurl,time_cut_b)
    j_struct_sync=j_cut_sync[cut_struct_idxs[i][0]:cut_struct_idxs[i][1],:]
    
    #determine structures for multispacecraft techniques
    bad_struct,crossing_time=msc.structure_crossing(b_field_struct_sync,
                                                  time_struct_b,data_gap_time)
    if(bad_struct): #not able to do multispacecraft techniques
        continue

    #do MDD analysis
    all_eigenvals,all_eigenvecs,avg_eigenvecs=msc.MDD(b_field_struct_sync,
                                                      rad_struct_sync)
    dims_struct,multidiml,D_struct,junk=msc.structure_diml(all_eigenvals)

    #calculate ion and electron velocities normal to structure (not in invariant dir.) 
    #and check dimensionality
    print(i)
    type_str='undefined'
    vi_norm=np.zeros(len(time_struct_b))
    ve_norm=np.zeros(len(time_struct_b))
    avg_guide_field=0
    b_mdd=np.empty_like(b_struct_bary) #per each eigenvector in timeseries
    b_mdd_avg=np.empty_like(b_struct_bary) #uses average eigenvectors
    if(dims_struct[2]):
        type_str="3D structure"
        for n in range(len(time_struct_b)):
            vi_norm[n]=np.linalg.norm(vi_struct_bary[n,:])
            ve_norm[n]=np.linalg.norm(ve_struct_bary[n,:])
            b_mdd[n,:]=np.transpose(all_eigenvecs[n,:,:]) @ b_struct_bary[n,:]
            b_mdd_avg[n,:]=np.transpose(avg_eigenvecs) @ b_struct_bary[n,:]
            avg_guide_field=avg_guide_field+b_mdd[n,2]/len(time_struct_b)
    elif(dims_struct[1]):
        type_str="2D structure"
        for n in range(len(time_struct_b)):
            vi_mdd=np.transpose(all_eigenvecs[n,:,:]) @ vi_struct_bary[n,:]
            vi_norm[n]=np.sqrt(vi_mdd[0]*vi_mdd[0]+vi_mdd[1]*vi_mdd[1])
            ve_mdd=np.transpose(all_eigenvecs[n,:,:]) @ ve_struct_bary[n,:]
            ve_norm[n]=np.sqrt(ve_mdd[0]*ve_mdd[0]+ve_mdd[1]*ve_mdd[1])
            b_mdd[n,:]=np.transpose(all_eigenvecs[n,:,:]) @ b_struct_bary[n,:]
            b_mdd_avg[n,:]=np.transpose(avg_eigenvecs) @ b_struct_bary[n,:]
            avg_guide_field=avg_guide_field+b_mdd[n,2]/len(time_struct_b)
    elif(dims_struct[0]):
        type_str="1D structure"
        for n in range(len(time_struct_b)):
            vi_mdd=np.transpose(all_eigenvecs[n,:,:]) @ vi_struct_bary[n,:]
            vi_norm[n]=abs(vi_mdd[0])
            ve_mdd=np.transpose(all_eigenvecs[n,:,:]) @ ve_struct_bary[n,:]
            ve_norm[n]=abs(ve_mdd[0])
            b_mdd[n,:]=np.transpose(all_eigenvecs[n,:,:]) @ b_struct_bary[n,:]
            b_mdd_avg[n,:]=np.transpose(avg_eigenvecs) @ b_struct_bary[n,:]
            avg_guide_field=avg_guide_field+b_mdd[n,2]/len(time_struct_b)
    if(multidiml):
        type_str=type_str+" with some smaller-dimensional qualities"



    #do STD analysis
    '''
    Todo: 
        -determine whether the STD analysis was good
    '''
    velocs,optimal=msc.STD(b_field_cut_sync,time_cut_b,cut_struct_idxs[i],
                               rad_struct_sync,all_eigenvals,all_eigenvecs,
                               dims_struct,min_crossing_height)
    vtot=np.linalg.norm(velocs,axis=1)
    avg_vtot=np.average(vtot)
    
    #calculate other properties of the structure (kind, size, etc.)
    #calculate the curvature of the magnetic field
    curvature_struct=msc.curvature(b_field_struct_sync,rad_struct_sync,
                                   b_struct_bary)
    total_curvature_struct=np.linalg.norm(curvature_struct,axis=1)
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
    MMS_structures.append(Structure(type_dict[type_flag],
                                                crossing_size,crossing_size_de,
                                                crossing_size_dp,
                                                avg_guide_field,avg_vtot,
                                                avg_vtot_e,avg_vtot_p))
    
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
        #plot it 
        mpl.rcParams.update(mpl.rcParamsDefault) #restores default plot style
        plt.rcParams.update({'figure.autolayout': True}) #plot won't overrun 
        gridsize=(6,2)
        fig=plt.figure(figsize=(16,12)) #width,height
        ax1=plt.subplot2grid(gridsize,(0,0))
        ax2=plt.subplot2grid(gridsize,(1,0))   
        ax3=plt.subplot2grid(gridsize,(2,0)) 
        ax4=plt.subplot2grid(gridsize,(3,0))
        ax5=plt.subplot2grid(gridsize,(4,0))
        ax6=plt.subplot2grid(gridsize,(5,0))
        ax6.axis('off')
        ax7=plt.subplot2grid(gridsize,(0,1))
        ax8=plt.subplot2grid(gridsize,(1,1))
        ax9=plt.subplot2grid(gridsize,(2,1))
        ax10=plt.subplot2grid(gridsize,(3,1))
        ax11=plt.subplot2grid(gridsize,(4,1))
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
        mmsp.tseries_plotter(fig,ax7,time_cut_b,j_cut_sync[:,1],j_labels,
                             lims=[min(time_cut_b),max(time_cut_b)]) #plot jy
        #plot the structure velocities
        for j in range(3):
            mmsp.tseries_plotter(fig,ax8,time_struct_b,
                                 velocs[:,j],
                                 labels=veloc_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=veloc_legend[j])
        mmsp.tseries_plotter(fig,ax8,time_struct_b,
                                 vtot,
                                 labels=veloc_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=veloc_legend[3]) 
        #plot comparison of total normal velocities to ion/electron normal velocities
        mmsp.tseries_plotter(fig,ax9,time_struct_b,
                                 vtot,
                                 labels=vcompare_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=vcompare_legend[0])
        mmsp.tseries_plotter(fig,ax9,time_struct_b,
                                 vi_norm,
                                 labels=vcompare_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=vcompare_legend[1])   
        mmsp.tseries_plotter(fig,ax9,time_struct_b,
                                 ve_norm,
                                 labels=vcompare_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=vcompare_legend[2])
        #plot densities
        mmsp.tseries_plotter(fig,ax10,time_struct_b,
                                 ni_struct_bary,
                                 labels=n_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=n_legend[0])  
        mmsp.tseries_plotter(fig,ax10,time_struct_b,
                                 ne_struct_bary,
                                 labels=n_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=n_legend[1])  
        #plot curvature components
        for j in range(3):
            mmsp.tseries_plotter(fig,ax11,time_struct_b,
                                     curvature_struct[:,j],
                                     labels=curvature_labels,
                                     lims=[min(time_cut_b),max(time_cut_b)],
                                     legend=curvature_legend[j]) 
        #plot total curvature 
        mmsp.tseries_plotter(fig,ax11,time_struct_b,
                                 total_curvature_struct,
                                 labels=curvature_labels,
                                 lims=[min(time_cut_b),max(time_cut_b)],
                                 legend=curvature_legend[3])            
        #add horizontal and vertical lines to plot (crossing + extent)
        mmsp.line_maker([ax1,ax2,ax3,ax4,ax5,ax7,ax8,ax9,ax10,ax11],
                        time=crossing_time,edges=crossing_struct_times[i],
                        horiz=0.)
         #add categorization information to plot
        ax6.text(0.5,0.5,type_str+'\n'+jy_sign_label+v_sign_label+ \
                 crossing_sign_label+crossing_size_label+crossing_dp_label+ \
                 crossing_de_label,wrap=True,
                 transform=ax6.transAxes,fontsize=16,ha='center',va='center')
        fig.savefig(os.path.join(timeseries_out_directory,'MMS'+'_'+ \
                                plot_out_name+str(i)+".png"), 
                                bbox_inches='tight')
        plt.close(fig="all")                                   

""" STATISTICAL PART """
''' make bar chart of the different types of structures '''
fig_bar,ax_bar=plt.subplots()
mmsp.bar_charter(ax_bar,MMS_structure_counts,['Types of structures seen by MMS',
                                         'Type of structure',
                                         'Number of instances']) 
fig_bar.savefig(os.path.join(statistics_out_directory,
                             "types_bar_chart"+".png"),bbox_inches='tight')
plt.close(fig='all')
''' make histograms of the x-lengths of all structures''' 
structure_kinds=type_dict.values()
mmsp.msc_structure_hist_maker(MMS_structures,"size",hists_out_directory,nbins,
                          structure_kinds)   
mmsp.msc_structure_hist_maker(MMS_structures,"size",hists_out_directory,
                          nbins_small,structure_kinds, log=True)
''' make histograms of the normalized x-lengths of all structures'''
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
''' make histograms of the guide field strengths of all structures'''
mmsp.msc_structure_hist_maker(MMS_structures,'guide_field',hists_out_directory,
                          nbins_small,structure_kinds)   
''' make histograms of the velocities of all structures ''' 
mmsp.msc_structure_hist_maker(MMS_structures,'normal_speed',
                              hists_out_directory,nbins,structure_kinds) 
mmsp.msc_structure_hist_maker(MMS_structures,'normal_speed',
                              hists_out_directory,nbins_small,structure_kinds,
                              log=True) 
''' make histograms of the normalized velocities of all structures ''' 
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
''' make scatter plot of guide field strength vs structure size '''
mmsp.msc_structure_scatter_maker(MMS_structures,'size','guide_field',
                             scatters_out_directory,structure_kinds)  
''' make scatter plot of normal speed vs structure size '''
mmsp.msc_structure_scatter_maker(MMS_structures,'size','normal_speed',
                             scatters_out_directory,structure_kinds)
''' make scatter plot of normalized speeds vs normalized structure size '''
mmsp.msc_structure_scatter_maker(MMS_structures,'electron_normalized_size',
                                 'electron_normalized_speed',
                                 scatters_out_directory,structure_kinds)
mmsp.msc_structure_scatter_maker(MMS_structures,'ion_normalized_size',
                                 'ion_normalized_speed',
                                 scatters_out_directory,structure_kinds)
''' make scatter plot comparing ion-normalized and electron-normalized attributes'''
mmsp.msc_structure_scatter_maker(MMS_structures,'electron_normalized_size',
                                 'ion_normalized_size',
                                 scatters_out_directory,structure_kinds)
mmsp.msc_structure_scatter_maker(MMS_structures,'electron_normalized_speed',
                                 'ion_normalized_speed',
                                 scatters_out_directory,structure_kinds)
    
#check how long the code took to run
end=time.time()
print("Code executed in "+str(dt.timedelta(seconds=end-start))) 

#To-do list:
#TODO: improve mechanism used to determine what time section to do the multi-spacecraft tech on
    #right now just what MMS1 says is the structure   
    #may have to do for now
#TODO: optimization- list appends are faster than numpy concatenates, so 
    #read in data as list of numpy arrays and then concatenate it all along the right axis
    #at the end for faster code
