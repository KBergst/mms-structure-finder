# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 13:28:44 2019

Home of functions to perform various multispacecraft techniques on MMS data
@author: kbergste
"""

import numpy as np
import datetime as dt
import mmstimes as mt
import mmsstructs as ms

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

def structure_crossing(b_fields,times,datagap):
    '''
    A multi-spacecraft approach to determining the average crossing time
    Inputs:
        b_fields- a dictionary with four values, each consisting of an array
            of b field data from a particular spacecraft of form (datlength,3)
        times- timestamps for the b_fields given (should be already synced)
        datagap- amount of time to classify something as a data gap

    Outputs:
        bad_struct- a boolean indicating whether the structure is good or not
    '''
    avg_crossing_time=0
    crossing_times=np.array([])
    
    bad_struct=False #True if all 4 spacecraft do not see the structure
    #compute average crossing time
    for b in b_fields.values():
        is_crossing=ms.find_crossings(b[:,2],
                                      times,datagap)
        if len(is_crossing) is 0: #no crossing for this satellite
            bad_struct=True
            break
        crossing_times=np.concatenate((crossing_times,times[is_crossing])) #add all crossings to list
    #find average crossing time
    if not bad_struct:
        avg_crossing_time=mt.datetime_avg(crossing_times)

    return bad_struct,avg_crossing_time  

def barycentric_vectors(spacecrafts_coords):
    '''
    Takes the positions of a tetrahedra of spacecraft (time synced) and
    calculates the reciprocal vectors for each spacecraft
    Inputs:
        spacecrafts_coords- a dictionary with four values, each consisting of 
            an array of spacecraft coordinates of form (datlength,3)
    Outputs:
        k_vec- a dictionary with four values, each consisting of an array of 
            reciprocal vectors of the form (datlength,3)
        
    '''
    k_vec={}
    
    #calculate reciprocal vectors
    scs=list(spacecrafts_coords.keys())
    r_scs=list(spacecrafts_coords.values()) #in python 3.7 and up this should keep the same ordering
    for i in range(len(r_scs)):
        #cyclic permutation of spacecraft
        j=(i+1) %4
        k=(i+2) %4
        l=(i+3) %4
        #determination of spacecraft separation vectors
        r_jk=r_scs[k]-r_scs[j]
        r_jl=r_scs[l]-r_scs[j]
        r_ji=r_scs[i]-r_scs[j]
        #calculation reciprocal vector for spacecraft scs[i]
        r_jk_x_r_jl=np.cross(r_jk,r_jl)
        denom=np.sum(r_ji*r_jk_x_r_jl,axis=1)
        k_vec[scs[i]]=r_jk_x_r_jl/denom[:,None] #allows division of the matrix by the vector in the sensical way

    return k_vec  

def spatial_gradient(vecs,spacecrafts_coords):
    '''
    Calculates a linear estimate of the spatial gradient of a vector quantity
    using the method outlined in Analysis Methods for Multi-Spacecraft Data, 
    Chapter 14.
    Inputs:
        vecs- a dictionary with four values, each consisting of an array
            of vector quantites from a particular spacecraft of form (datlength,3)
        spacecrafts_coords- a dictionary with four values, each consisting of 
            an array of spacecraft coordinates of form (datlength,3)    
    Outputs:
        grads- a list of the gradient tensors for the vector quantity given,
            at each timetag.
    '''
    #find reciprocal vectors
    k_vecs=barycentric_vectors(spacecrafts_coords)
    grads=[]

    scs=list(spacecrafts_coords.keys())
    #calculate the spatial gradient- likely a faster way exists, will need to see
    for n in range(len(spacecrafts_coords[scs[0]][:,0])): 
        grad=np.zeros((3,3))                   
        for i in range(len(scs)):
            #reshape the n'th vector for scs[i] for matrix multiplying
            vec=vecs[scs[i]][n,:]
            vec_T=vec.reshape(1,3)
            #add the spacecraft's contribution to the total spatial gradient
            k_vec=(k_vecs[scs[i]][n,:]).reshape(3,1)
            grad=grad+k_vec @ vec_T
        #add spatial gradient to list of spatial gradients
        grads.append(grad)          
      
    return grads

def time_deriv(array,times,deriv_window,dt_num):
    '''
    Calculates the centered-difference time derivative of value(s)
    Inputs:
        array- array to be time-differenced of form (datlength,n) for n the 
            number of different scalar quantities to take derivatives of
        times- the timeseries of the array of form (datlength)
        deriv_window- two-element list of the beginning and end indices of 
                the list of elements to have time-differencing done to
        dt_num- number of data points to each side to do the central differencing
            over (may want to do weighted average in the future? unclear)
    Outputs:
        derivs- the time derivative of array, in units/second, over the
            indexes specified by deriv_window
    '''
    list_derivs=[]
    for idx in range(deriv_window[0],deriv_window[1]):
        time_diff=(times[idx+dt_num]-times[idx-dt_num]).total_seconds()
        array_diff=array[idx+dt_num,:]-array[idx-dt_num,:]
        tmp=array_diff/time_diff
        deriv=tmp.reshape(1,len(tmp))
        list_derivs.append(deriv)
    derivs=np.concatenate(list_derivs,axis=0)
    
    return derivs
      
def MDD(b_fields,spacecrafts_coords):
    '''
    Does a Minimum Directional Derivative (MDD) analysis on magnetic field data
    (see Shi et al. 2005 for reference)
    Inputs:
        b_fields- a dictionary with four values, each consisting of an array
            of magnetic field vectors from a particular spacecraft of form 
            (datlength,3)
        spacecrafts_coords- a dictionary with four values, each consisting of 
            an array of spacecraft coordinates of form (datlength,3)
    Outputs:
        all_eigenvals-arrays of the three eigenvalues from the analysis
            ordered from largest to smallest in shape (datlength,3)
        all_eigenvecs- array of arrays of the three eigenvectors in shape
            (datlength,3,3)
    '''
    b_grads=spatial_gradient(b_fields,spacecrafts_coords)
    
    all_eigenvals=np.transpose(np.array([[],[],[]]))
    list_eigenvals=[]
    list_eigenvecs=[]
    for grad_b in b_grads:
        L=grad_b @ np.transpose(grad_b)
        
        #compute eigenvectors and values of Mb, and sort them by decreasing magnitude
        tmp1,tmp2=np.linalg.eig(L) #column tmp2[:,i] is the eigenvector
        idx=tmp1.argsort()[::-1] #default sort is small to big, so need to reverse
        eigenvals=tmp1[idx].reshape(1,3)
        eigenvecs=tmp2[:,idx]
            
        eigenvecs=eigenvecs.reshape((1,3,3))    
            
        list_eigenvals.append(eigenvals)
        list_eigenvecs.append(eigenvecs)

    #convert eigenvecs,eigenvals to numpy array
    all_eigenvals=np.concatenate(list_eigenvals,axis=0)
    all_eigenvecs=np.concatenate(list_eigenvecs,axis=0)
    #flip all eigenvectors for the structure into a standardized direction
    datlength=np.size(all_eigenvecs,axis=0)
    center_eigenvecs=all_eigenvecs[datlength//2,:,:]
    for n in range(datlength): #probably a faster way to do than nested for loops
        for i in range(3):
            if np.dot(center_eigenvecs[:,i],all_eigenvecs[n,:,i]) < 0:
                all_eigenvecs[n,:,i]=-1*all_eigenvecs[n,:,i]
    
    return all_eigenvals, all_eigenvecs    

def structure_diml(mdd_eigenvals):
    '''
    determines the dimensionality of a structure based on its eigenvalues from 
    MDD analysis and also returns its invariant directions (if applicable).
    
    This function is ESPECIALLY open to change, as there are many different
    conventions for interpreting the results of MDD, and none seem to have
    consensus. I'll probably try a bunch of different ones to see what I like.
    Inputs:
        mdd_eigenvals-arrays of the three eigenvalues from the analysis
            ordered from largest to smallest in shape (datlength,3)
    Outputs:
        dims- list of three dimensions- true if the code thinks it is that
            dimensionality and false if the code thinks it is not that 
            dimensionality.
        multi_diml- returns True if more than one element of dims is True.
        D_avg- the average D1,2,3 values calculated (as per Rezeau et al. 2018)
        [D_2Davg,D_3Davg]- list containing the selection parameters from the 
            Sun et al. 2019 method. For troubleshooting the method, etc.
    '''
    dims=[False,False,False] #dims[0],[1],[2] refer to 1,2, and 3d respectively
    # Rezeau et al. 2018 method (might fold all D values into an array at some point)
    D=[]
    D_avg=[]
    D.append((mdd_eigenvals[:,0]-mdd_eigenvals[:,1])/mdd_eigenvals[:,0])
    D.append((mdd_eigenvals[:,1]-mdd_eigenvals[:,2])/mdd_eigenvals[:,0])
    D.append(mdd_eigenvals[:,2]/mdd_eigenvals[:,0])
    D_avg.append(np.average(D[0]))
    D_avg.append(np.average(D[1]))
    D_avg.append(np.average(D[2]))
    
    #Sun et al. 2019 method
    D_2D=np.sqrt(mdd_eigenvals[:,1]/mdd_eigenvals[:,0])
    D_3D=np.sqrt(mdd_eigenvals[:,2]/mdd_eigenvals[:,0])
    D_2D_avg=np.average(D_2D)
    D_3D_avg=np.average(D_3D)
    
    #combine for the method:
    if (D_avg[0]>0.7 or D_2D_avg < 0.4):
        dims[0]=True
    if (D_avg[1]>0.7 or (D_2D_avg > 0.4 and D_3D_avg < 0.4)):
        dims[1]=True
    if (D_avg[2]>0.7 or D_3D_avg > 0.4):
        dims[2]=True
        
    if sum(dims)>1:
        multi_diml=True
    else:
        multi_diml=False
        
    return dims,multi_diml,D_avg,[D_2D_avg,D_3D_avg]

def STD(b_fields,times,struct_idxs,spacecrafts_coords,mdd_eigenvals,
        mdd_eigenvecs,dims,b_err,datarate=1/128):
    '''
    determines the velocity of 1D, 2D and 3D structures using the
    Spatio-Temporal Difference (STD) method, as outlined in Shi et al. 2006.
    Inputs:
        b_fields- a dictionary with four values, each consisting of an array
            of magnetic field vectors from a particular spacecraft of form 
            (datlength,3). contains information from the entire window
        times-timeseries for the b-field data
        struct_idxs- two-element list of the beginning and end indices of 
            the structure within the b_field data.
        spacecrafts_coords- a dictionary with four values, each consisting of 
            an array of spacecraft coordinates of form (datlength,3)
        mdd_eigenvals-arrays of the three eigenvalues from the analysis
            ordered from largest to smallest in shape (datlength,3)
        mdd_eigenvecs- list of arrays of the three eigenvectors from the MDD 
            analysis at each point.
        dims-list of three dimensions- true if the code thinks it is that
            dimensionality and false if the code thinks it is not that 
            dimensionality. [1D,2D,3D] 
        b_err-expected error threshold for the magnetic field data.
        datarate-expected time cadence of data. Default 1/128 s
    Outputs:
        normal_veloc- contains the normal velocity in GSM coordinates
            a numpy array of shape (datlength,3)
        optimal- returns False if the time difference wasn't able to fit the
            criteria outlined in Shi et al. 2006
    '''
    optimal=True
    db_dt_avg=np.zeros_like(mdd_eigenvals)
    veloc=np.zeros_like(mdd_eigenvals) #returns zero if the velocity shouldn't be determined
    normal_veloc=veloc #will contain the normal velocity in GSM coordinates
    #calculate the average total time derivative in nT/s
    b_field_struct={}
    for sc in b_fields.keys():
        dt_num=1
        db_dt=time_deriv(b_fields[sc],times,struct_idxs,dt_num) #in nT/s
        while (b_err/np.average(np.linalg.norm(db_dt,axis=1)) > \
               dt_num*datarate): #need to redo time derivative with larger sectioning
            dt_num=int(b_err/np.average(np.linalg.norm(db_dt,axis=1))/ \
                       datarate)+1
            if(dt_num >= struct_idxs[0]-1): #unable to do a large enough window to average
                print("time differencing unable to meet optimal criteria")
                optimal=False
                break
            db_dt=time_deriv(b_fields[sc],times,struct_idxs,dt_num) #in nT/s
        db_dt_avg=db_dt_avg+db_dt/len(b_fields.values())
        #section the b-field data to have only the part which has associated derivatives
        b_field_struct[sc]=b_fields[sc][struct_idxs[0]:struct_idxs[1],:]
        
    #calculate the spatial gradient of the magnetic field
    grad_B=spatial_gradient(b_field_struct,spacecrafts_coords)
    
    #do the STD analysis for each spacecraft
    if dims[0]: #only take one dimension
        for n in range(struct_idxs[1]-struct_idxs[0]):
            dbdt=db_dt_avg[n,:]
            lhs= dbdt @ np.transpose(grad_B[n]) @ mdd_eigenvecs[n,:,:]
            veloc[n,0]=-1*lhs[0]/mdd_eigenvals[n,0]
            normal_veloc[n,:]= mdd_eigenvecs[n,:,:] @ veloc[n,:]
    elif dims[1]: #only take two dimensions
        for n in range(struct_idxs[1]-struct_idxs[0]):
            dbdt=db_dt_avg[n,:]
            lhs= dbdt @ np.transpose(grad_B[n]) @ mdd_eigenvecs[n,:,:]
            veloc[n,0]=-1*lhs[0]/mdd_eigenvals[n,0]
            veloc[n,1]=-1*lhs[1]/mdd_eigenvals[n,1]
            normal_veloc[n,:]= mdd_eigenvecs[n,:,:] @ veloc[n,:]
    else: #do all three dimensions
        for n in range(struct_idxs[1]-struct_idxs[0]):
            dbdt=db_dt_avg[n,:]
            lhs= dbdt @ np.transpose(grad_B[n]) @ mdd_eigenvecs[n,:,:]
            veloc[n,0]=-1*lhs[0]/mdd_eigenvals[n,0]
            veloc[n,1]=-1*lhs[1]/mdd_eigenvals[n,1]
            veloc[n,2]=-1*lhs[2]/mdd_eigenvals[n,2]
            normal_veloc[n,:]= mdd_eigenvecs[n,:,:] @ veloc[n,:]
    
    return normal_veloc,optimal
    

    