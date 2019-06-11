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

def electron_veloc_x(j,time_j,vi,ni,time_ni,ne,time_ne):
    '''
    Used to calculate the x component of the electron velocity from curlometer 
    current, ion velocity, and electron and ion densities.
    Inputs:
        j- array with the curlometer current in microAmps
        time_j- the timeseries for the curlometer current array (nanosecs)
        vi- the ion velocity in km/s
        ni- the ion density in cm^-3
        time_ni- the timeseries for both the ion density and velocity (nanosecs)
        ne- the electron density in cm^-3
        time_ne- the timeseries for electron density (nanosecs)
    Outputs:
        vex_arr- the calculated electron velocity as a numpy array in km/s
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
        vex_list.append(vex_time) 
    vex_arr=np.array(vex_list)
    return vex_arr
