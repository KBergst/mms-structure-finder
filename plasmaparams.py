# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 13:04:00 2019

@author: kbergste

module for functions that calculate plasma parameters e.g. plasma frequency,
skin depth
"""

import scipy.constants as const
import numpy as np

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
