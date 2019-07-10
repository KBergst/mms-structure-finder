# -*- coding: utf-8 -*-
"""
Created on Wed July 5 09:34:54 2019

@author: matthewchen37

Definitions for functions that are used for fitting 
"""


import math
import scipy as sp

#TODO: Write a function that normalizes the data 

#Turns (x,y,z) coordinates into (r, theta, z) coordinates 
def RectToCylindrical(RectArray):   
    RectArray = normalize(RectArray)
    row, column = len(RectArray), len(RectArray[0]) #Gets the Height and Width of Array for conversion
    CylindArray = [[0 for x in range(column)] for y in range(row)] #Initializes the return array as empty
    for i in range(row): #Each row is a 3 dimensional vector, this loops goes down each row
        x = RectArray[i][0]
        y = RectArray[i][2]
        r = math.sqrt((x ** 2) + (y ** 2)) # finds radial distance or B-Azimuthal
        if x > 0:
            theta = math.atan(y/x) #Finds angle theta based off of x and y vectors, theta is in RADIANS
        elif x == 0:
            theta = 0
        else:
            theta = -math.asin(y/r) + math.pi
        CylindArray[i][0] = r #Store radial distance in first position --> Azimuthal
        CylindArray[i][2] = theta #Stores angular direction in second position --> Azimuthal 
        CylindArray[i][1] = RectArray[i][1] #Z position left untouched, stored in 3rd position --> Axial Coordinate
     
    return CylindArray 

def getAzimuthal(maxVar, minVar):
    B_azi = math.sqrt((maxVar ^ 2) + (minVar ^ 2))
    return B_azi


def modelFluxRope(impact_param): #Mmodel flux rope based off of Smith et al., 2017
    ''' We assume that B0 and the Helicity are equal to 1 so that they are normalized'''
    B0 = 1
    H = 1
    alpha = 2.4048 #From paper, constant alpha makes the flux rope linear
    B_axial = B0 * sp.special.jv(0, alpha * impact_param)
    B_azimuthal = B0 * H * sp.special.jv(1, alpha * impact_param)
    return B_axial,B_azimuthal




def normalize(oldArray): #makes all vectors a ratio of the largest magnitude vector 
    maxMagnitude = 0
    row, column = len(oldArray), len(oldArray[0]) #Gets the Height and Width of Array for conversion
    normalized = [[0 for x in range(column)] for y in range(row)] #Initializes the return array as empty
    for x in range(len(oldArray)): #searches for maximum magnitude
        magnitude = math.sqrt((oldArray[x][0] ** 2) + (oldArray[x][1] ** 2) + (oldArray[x][2] ** 2))    
        if magnitude > maxMagnitude:
            maxMagnitude = magnitude
    for x in range(len(oldArray)): #converts all values into ratio of maximum magnitude
        magnitude = math.sqrt(oldArray[x][0] ** 2 + (oldArray[x][1] ** 2) + (oldArray[x][2] ** 2))   
        ratio = magnitude / maxMagnitude
        normalized[x][0] = oldArray[x][0] * (ratio/magnitude)
        normalized[x][1] = oldArray[x][1] * (ratio/magnitude)
        normalized[x][2] = oldArray[x][2] * (ratio/magnitude)
    return normalized 
        
    

def chisquared1(RectArray): #first chi-squared test as defined by Smith et al., 2017
    CylindArray = RectToCylindrical(RectArray) 
    #not sure if this is correct -- made a lot of assumptions
    impactParameter = 0
    minChiSquared = 0
    chiSquaredValue = 0 
    for y in range(0, 95, 1):
        imp = y / 100
        B_axial, B_azimuthal = modelFluxRope(y)
        for x in range(len(RectArray)):
            chiSquaredValue += ((CylindArray[x][0]- B_azimuthal) ** 2) + ((CylindArray[0][1] - B_axial) ** 2)
            '''for  the model equation does it only give magnitudes in the cylindrical coordinate system?
               in the paper is the chi squared only comparing the magnitudes of the magnetic fields in the azimuthal direction?'''
        chiSquaredValue = chiSquaredValue / len(RectArray)
        if imp == 0:
            impactParameter = 0
            minChiSquared = chiSquaredValue
        elif chiSquaredValue < minChiSquared:
            minChiSquared = chiSquaredValue
            impactParameter = imp 
    print("The impact parameter is" + str(impactParameter))
    if impactParameter > 0.5:
        print ("false")
        minChiSquared = False
        return minChiSquared, impactParameter
    else:
        return minChiSquared, impactParameter 


def getDist(imp_param,numOfDataPoints):
    l = math.sqrt(1 - (imp_param ** 2)) #half of the chord
    lengthOfChord = 2 * l
    dist = lengthOfChord / numOfDataPoints
    return l, dist


def getTheta(imp_param, numOfDataPoints, index, l, dist):
    theta = 0
    half = math.ceil(numOfDataPoints / 2)
    if (index > half): #right side
        theta = math.atan(((index - half) * dist )/imp_param)
    else:
        theta = math.atan((l - (dist * index))/imp_param)
    return theta
        
    
def getComponents(theta, b_azi):
    b_model_min = b_azi * math.cos(theta)
    b_model_max = b_azi * math.sin(theta)
    return b_model_min, b_model_max

    
def chiSquared2(RectArray, imp_Param): #No conversion is needed for this one
    RectArray = normalize(RectArray)
    numOfData = len(RectArray)
    chiSquare = 0
    b_axial, b_azim = modelFluxRope(imp_Param)
    l, dist = getDist(imp_Param, numOfData)
    for x in range(numOfData):
        theta = getTheta(imp_Param, numOfData, x, l, dist)
        b_model_min, b_model_max = getComponents(theta, b_azim)
        chiSquare += ((RectArray[x][0] - b_model_max) ** 2) + ((RectArray[x][1] - b_axial) ** 2) + ((RectArray[x][2] - b_model_min)  ** 2)
    chiSquare = (3 * numOfData) - 4
    return chiSquare



        
    
    
    