# -*- coding: utf-8 -*-
"""
Created on Wed July 5 09:34:54 2019

@author: matthewchen37

Definitions for functions that are used for fitting 
"""


import math
import scipy
#TODO: Write a function that normalizes the data 

#Turns (x,y,z) coordinates into (r, theta, z) coordinates 
def RectToCylindrical(RectArray):   
    row, column = len(RectArray), len(RectArray[0]) #Gets the Height and Width of Array for conversion
    CylindArray = [[0 for x in range(column)] for y in range[row]] #Initializes the return array as empty
    for i in range(row): #Each row is a 3 dimensional vector, this loops goes down each row
        x = RectArray[i][0]
        y = RectArray[i][2]
        r = math.sqrt((x ^ 2) + (y ^ 2)) # finds radial distance or B-Azimuthal
        theta = math.atan(y/x) #Finds angle theta based off of x and y vectors, theta is in RADIANS
        CylindArray[i][0] = r #Store radial distance in first position --> Azimuthal
        CylindArray[i][2] = theta #Stores angular direction in second position --> Azimuthal 
        CylindArray[i][1] = RectArray[i][1] #Z position left untouched, stored in 3rd position --> Axial Coordinate
     
    return CylindArray 



def modelFluxRope(impact_param): #Mmodel flux rope based off of Smith et al., 2017
    ''' We assume that B0 and the Helicity are equal to 1 so that they are normalized'''
    B0 = 1
    H = 1
    alpha = 2.4048 #From paper, constant alpha makes the flux rope linear
    B_axial = B0 * J0 * scipy.jv(0, alpha * impact_param)
    B_azimuthal = B0 * H * scipy.jv(1, alpha * impact_param)
    return B_axial,B_azimuthal




def normalize(oldArray): #makes all vectors a ratio of the largest magnitude vector 
    index = 0
    maxMagnitude = 0
    row, column = len(oldArray), len(oldArray[0]) #Gets the Height and Width of Array for conversion
    newArray = [[0 for x in range(column)] for y in range[row]] #Initializes the return array as empty
    for x in range(len(oldArray)):
        magnitude = math.sqrt(oldArray[x][0] ^ 2 + (oldArray[x][1] ^ 2) + (oldArray[x][2] ^ 2))    
        if( magnitude > maxMagnitude)
            index = x
            maxMagnitude = magnitude
    for x in range(len(oldArray)):
        magnitude = math.sqrt(oldArray[x][0] ^ 2 + (oldArray[x][1] ^ 2) + (oldArray[x][2] ^ 2))    
        ratio = magnitude / maxMagnitude
        newArray[x][0] = oldArray[x][0] * ratio
        newArray[x][1] = oldArray[x][1] * ratio
        newArray[x][2] = oldArray[x][2] * ratio
    return newArray 
        
    

def chisquared1(RectArray): #first chi-squared test as defined by Smith et al., 2017
    CylindArray = RectToCylindrical(RectArray)
    for x in range(len(CylindArray)):
        for y in range(0, 0.95, 0.01):
            
    