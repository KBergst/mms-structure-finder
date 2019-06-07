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
import mmstimes as mt
import plasmaparams as pp
import mmsdata as md
import mmsarrays as ma
import mmsstructs as ms

#canned packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime as dt
import sys #for debugging
#from pympler.tracker import SummaryTracker #for tracking memory usage
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
REPLOT=1 #chooses whether to regenerate the graphs or not

###### CLASS DEFINITIONS ######################################################
class Structure:
    
    #initializer
    def __init__(self,kind,size):
        self.kind=kind
        self.size=size
###### MAIN ###################################################################
#ensuring that the needed output directories exist
mmsp.directory_ensurer(os.path.join(path,plot_out_directory))
mmsp.directory_ensurer(os.path.join(path,statistics_out_directory))
mmsp.directory_ensurer(os.path.join(path,scales_out_directory))

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
    
    b_list=md.filenames_get(os.path.join(path,data_dir,M,bfield_names_file))
    dis_list=md.filenames_get(os.path.join(path,data_dir,M,dis_names_file))
    des_list=md.filenames_get(os.path.join(path,data_dir,M,des_names_file))
    j_list=md.filenames_get(os.path.join(path,j_names_file))

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
        TT_time_tmp,temp=md.get_cdf_var(b_file,['Epoch',
                                             'mms'+M+'_fgm_b_gsm_brst_l2'])
        b_field_tmp=temp.reshape(temp.size//4,4) #// returns integer output
        b_field=np.concatenate((b_field,b_field_tmp),axis=0)
        TT_time_b=np.concatenate((TT_time_b,TT_time_tmp))
        #read and process the ni,vi data
        TT_time_tmp,ni_tmp,ni_err_tmp,temp=md.get_cdf_var(dis_file,
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
        TT_time_tmp,ne_tmp,ne_err_tmp,temp=md.get_cdf_var(des_file,
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
            time_reg_j_tmp,j_curl_tmp=md.import_jdata(j_file)
            j_curl=np.concatenate((j_curl,j_curl_tmp),axis=0)
            time_reg_jcurl=np.concatenate((time_reg_jcurl,time_reg_j_tmp))
    TT_time_j=mt.datetime2TTtime(time_reg_jcurl) #time to nanosecs for interpolating
    time_reg_b=np.array(mt.TTtime2datetime(TT_time_b)) #time as datetime obj np arr
    time_reg_ni=np.array(mt.TTtime2datetime(TT_time_ni))
    time_reg_ne=np.array(mt.TTtime2datetime(TT_time_ne))
    by=b_field[:,1]
    bz=b_field[:,2]
    jy=j_curl[:,1]
    vex_fpi=ve_fpi[:,0]
    vix=vi[:,0]
    ne_smooth=ma.boxcar_avg(ne,boxcar_width) #smooth the ne data to avoid zeroes
    ne_nozero=np.where(ne_smooth>ne_fudge_factor,ne_smooth,ne_fudge_factor)
    #roughly calculate electron velocity from curlometer
    vex=pp.electron_veloc_x(j_curl,TT_time_j,vi,ni,TT_time_ni,ne_nozero,
                         TT_time_ne)
    #calculate approximate electron and ion plasma frequencies and skin depths
    we=pp.plasma_frequency(ne_nozero,const.m_e)
    wp=pp.plasma_frequency(ni,const.m_p) #assuming all ions are protons (valid?)
    de=pp.inertial_length(we)
    dp=pp.inertial_length(wp)
    
    #locate crossings and their directions
    crossing_indices_bz=ms.find_crossings(bz,time_reg_b,data_gap_time)
    crossing_signs_bz=ms.find_crossing_signs(bz,crossing_indices_bz)
    crossing_indices_bz,crossing_signs_bz,max_indices,min_indices= \
                                                        ms.find_maxes_mins(bz,
                                                        crossing_indices_bz,
                                                        crossing_signs_bz,
                                                        extrema_width,
                                                        min_crossing_height)
    crossing_times=time_reg_b[crossing_indices_bz]
    #section the data and define structural extents
    crossing_windows=ms.section_maker(crossing_indices_bz,max_indices,min_indices,
                                   window_padding,len(bz))
    crossing_structs,crossing_struct_times=ms.structure_extent(
                                    crossing_indices_bz,time_reg_b,
                                    crossing_signs_bz,max_indices,min_indices,
                                    len(bz))
    #process each crossing
    for i in range(len(crossing_indices_bz)):
        '''Slicing and sectioning the various data arrays '''
        #slice b and b timeseries, set plotting limits
        time_b_cut=time_reg_b[crossing_windows[i][0]:crossing_windows[i][1]]
        bz_cut=bz[crossing_windows[i][0]:crossing_windows[i][1]] #for window
        by_cut=by[crossing_windows[i][0]:crossing_windows[i][2]]
        by_struct=by[crossing_structs[i][0]:crossing_structs[i][1]]
        time_b_struct=time_reg_b[crossing_structs[i][0]:crossing_structs[i][1]]
        plot_limits=[time_b_cut[0],time_b_cut[-1]] #data section
        #slice ni,vi and ni timeseries
        window_mask_ni=ma.interval_mask(time_reg_ni,plot_limits[0],plot_limits[1])
        struct_mask_ni=ma.interval_mask(time_reg_ni,time_b_struct[0],
                                     time_b_struct[1])
        time_ni_cut=time_reg_ni[window_mask_ni]
        ni_cut=ni[window_mask_ni]
        ni_err_cut=ni_err[window_mask_ni]
        vix_cut=vix[window_mask_ni]
        vix_fluct=ma.fluct_abt_avg(vix_cut) #for fluctuation plot
        #slice ne and ne timeseries
        window_mask_ne=ma.interval_mask(time_reg_ne,plot_limits[0],plot_limits[1])
        struct_mask_ne=ma.interval_mask(time_reg_ne,time_b_struct[0],
                                     time_b_struct[1])
        time_ne_cut=time_reg_ne[window_mask_ne]
        ne_cut=ne[window_mask_ne]
        ne_smooth_cut=ne_smooth[window_mask_ne]
        ne_err_cut=ne_err[window_mask_ne]  
        #slice ve from FPI timeseries
        vex_fpi_cut=vex_fpi[window_mask_ne]
        #slice j and j timeseries
        window_mask_j=ma.interval_mask(time_reg_jcurl,plot_limits[0],
                                    plot_limits[1])
        struct_mask_j=ma.interval_mask(time_reg_jcurl,time_b_struct[0],
                                    time_b_struct[-1])
        time_j_cut=time_reg_jcurl[window_mask_j] #for window
        jy_cut=jy[window_mask_j] #for window
        time_j_struct=time_reg_jcurl[struct_mask_j] #for structure
        jy_struct=jy[struct_mask_j] #for structure
        #slice ve curlometer timeseries (same as j)
        vex_cut=vex[window_mask_j] #for window
        vex_struct=vex[struct_mask_j] #for structure
        vex_fluct=ma.fluct_abt_avg(vex_struct) #for fluctuation plot
        #slice inertial lengths and average
        de_cut=de[window_mask_ne]
        de_cut_avg=np.average(de_cut)
        dp_cut=dp[window_mask_ni]
        dp_cut_avg=np.average(dp_cut)
        str_de_avg=f"{de_cut_avg:.1f}"  #string formatting
        str_dp_avg=f"{dp_cut_avg:.1f}"  #string formatting
        
        '''Additional calculation of relevant information '''
        #determine signs of vex and jy
        jy_sign,jy_qual=ma.find_avg_signs(jy_struct)
        vex_sign,vex_qual=ma.find_avg_signs(vex_struct)
        #determine the average By over structure and window (guide field approx)
        by_cut_avg=np.average(by_cut)
        by_struct_avg=np.average(by_struct)
        
        #determine crossing clasification and size and update counts:
        crossing_type,type_flag=ms.structure_classification(crossing_signs_bz[i],
                                                         vex_sign,vex_qual,
                                                         jy_sign,jy_qual,
                                                         quality_min)
        crossing_size=ms.structure_sizer([time_b_struct[0],time_b_struct[1]],
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
                larger_window=ms.larger_section_maker(crossing_windows[i],
                                                   window_scale_factor,
                                                   len(bz))   
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
        