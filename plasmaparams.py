# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:04:00 2019

@author: kbergste

module for functions that calculate plasma parameters e.g. plasma frequency,
skin depth
"""

import scipy.constants as const
import numpy as np
import scipy.interpolate as interp #for interpolating to different times
E_CHARGE_mC=const.e*1e6 #electron charge in microcoulombs

def plasma_frequency(density,mass,zeff=1):
    '''
    defines the plasma frequency in rad/s given density and mass
    Inputs:
        mass- expected in kg
        density- expected in cm^-3
        zeff- the effective charge of the particle (1 if proton or electron)
    Outputs:
        freq- the plasma frequency in rad/s
    '''
    charge=zeff*const.e
    density_m3=density*1e6 #cm^-3 to m^-3
    freq=np.sqrt(density_m3*charge*charge/mass/const.epsilon_0)
    return freq

def inertial_length(freq):
    '''
    defines the inertial length in km given the plasma frequency in rad/s
    for either electrons or some ion species
    Inputs:
        freq- the plasma frequency in rad/s
    Outputs:
        d- the inertial length in km
    '''
    c_km_s=const.c/1e3 #m/s to km/s
    d=c_km_s/freq
    return d

def beta(pressure,bfield):
    '''
    Use to calculate the average plasma beta given the pressure (in nPa) and the
    magnetic field (in nT). Pressure and bfield MUST be time synced!
    Inputs:
        pressure- the total pressure in nPa
        bfield- the magnetic field as a numpy array in nT,
            dimension (arrlength,3)
    Outputs:
        
    '''
    numerator=pressure*2*const.mu_0*1e9 #1e9 to make mu_0 the right units
    denominator=np.linalg.norm(bfield,axis=1)**2
    plasma_beta=numerator/denominator
    
    return plasma_beta

def electron_veloc(j,time_j,vi,ni,time_ni,ne,time_ne):
    '''
    Used to calculate the electron velocity from curlometer 
    current, ion velocity, and electron and ion densities.
    Inputs:
        j- array with the curlometer current in microAmps/m^2
        time_j- the timeseries for the curlometer current array (nanosecs)
        vi- the ion velocity in km/s
        ni- the ion density in cm^-3
        time_ni- the timeseries for both the ion density and velocity (nanosecs)
        ne- the electron density in cm^-3
        time_ne- the timeseries for electron density (nanosecs)
    Outputs:
        ve- the calculated electron velocity as a numpy array in km/s,
            dimension (arrlength,3)
    TODO: calculate all electron velocity components and remove the need to use
        lists by using the np.vstack method
    '''
    vi_interp=interp.interp1d(time_ni,vi,kind='linear',axis=0,
                              assume_sorted=True)
    ni_interp=interp.interp1d(time_ni,ni, kind='linear',
                              assume_sorted=True)
    ne_interp=interp.interp1d(time_ne,ne, kind='linear',
                              assume_sorted=True)
    ve_list=[]
#    vex_list_novi=[]
    for n,ttime in enumerate(time_j):
        tmp=(vi_interp(ttime)*ni_interp(ttime)*1e9* \
                  E_CHARGE_mC-j[n,:])/E_CHARGE_mC/(ne_interp(ttime))/1e9
        ve_time=tmp.reshape((1,3))
        ve_list.append(ve_time) 

    ve=np.concatenate(ve_list,axis=0)
    return ve

def j_dot_e(j,E,B):
    '''
    Finds the energy exchange between fields and particles, and in particular
    the quantities perpendicular and parallel to the magnetic field.
    Inputs:
        j- array with the curlometer current in microAmps/m^2
        E- array with electric field in mV/m
        B- array with magnetic field in nT
    Outputs:
        jE- total j dot E at each point in time
        jE_para- the parallel component of j dot E at each point in time
        jE_perp- the perpendicular component of j dot E at each point in time
    '''
    B_tot=np.linalg.norm(B,axis=1)
    jE_list=[]
    jE_para_list=[]
    jE_perp_list=[]
    for n in range(len(j[:,0])):        
        j_dot_E=np.dot(j[n,:],E[n,:])
        j_para=np.dot(j[n,:],B[n,:])/B_tot[n]
        E_para=np.dot(E[n,:],B[n,:])/B_tot[n]
        j_dot_E_para=j_para*E_para
        j_dot_E_perp=j_dot_E-j_dot_E_para
        
        jE_list.append([j_dot_E])
        jE_para_list.append([j_dot_E_para])
        jE_perp_list.append([j_dot_E_perp])
        
    jE=np.concatenate(jE_list)
    jE_para=np.concatenate(jE_para_list)
    jE_perp=np.concatenate(jE_perp_list)
    
    return jE,jE_para,jE_perp
    
    

