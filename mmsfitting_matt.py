# -*- coding: utf-8 -*-
"""
Created on Wed July 5 09:34:54 2019

@author: matthewchen37

Definitions for functions that are used for fitting 
"""


import math
from scipy import special as sp

#TODO: Write a function that normalizes the data 

#Turns (x,y,z) coordinates into (r, z, theta) coordinates 
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
    B0 = 1 #should it equal 1? this is the missing piece?
    H = 1
    alpha = 2.4048 #From paper, constant alpha makes the flux rope linear
    B_axial = B0 * sp.jv(0, alpha * impact_param)
    B_azimuthal = B0 * H * sp.jv(1, alpha * impact_param)
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
        
def normalize2(oldArray, impact_param): #maximmum magnitude is based off of the bessel function, which is a funciton of the impact parameter
    b_axi, b_azi = modelFluxRope(impact_param)
    maxMagnitude = math.sqrt((b_axi ** 2 ) + (b_azi **2))
    row, column = len(oldArray), len(oldArray[0]) #Gets the Height and Width of Array for conversion
    normalized = [[0 for x in range(column)] for y in range(row)] #Initializes the return array as empty
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
    isEven = False
    if len(RectArray) % 2 == 0:
        isEven = True
        
    for y in range(0, 95, 1):
        imp = y / float(100)
        #print("The chi-squared before turning to 0" + str(chiSquaredValue))
        chiSquaredValue = 0 
        for x in range(len(RectArray)):
            radialDist = getRadialDistance(imp, len(RectArray), x, isEven)
            B_axial, B_azimuthal = modelFluxRope(radialDist)
           # print("B-Axial: " + str(B_axial) + "B-azimuthal: " + str(B_azimuthal))
            chiSquaredValue += ((CylindArray[x][0] - B_azimuthal) ** 2) + ((CylindArray[0][1] - B_axial) ** 2)
            
            '''for  the model equation does it only give magnitudes in the cylindrical coordinate system?
               in the paper is the chi squared only comparing the magnitudes of the magnetic fields in the azimuthal direction?'''
        chiSquaredValue = chiSquaredValue / len(RectArray)
        if imp == 0.0:
            minChiSquared = chiSquaredValue
        elif chiSquaredValue < minChiSquared:
            minChiSquared = chiSquaredValue
            impactParameter = imp 
    print("The impact parameter is" + str(impactParameter))
    if impactParameter > 0.5:
        
        minChiSquared = False
        return minChiSquared, impactParameter
    else:
        return minChiSquared, impactParameter 

def getTheta(radialDist, s):
    if (s == 0):
        theta = 0
    else:
        theta = math.atan(radialDist / s)
    return theta
        
    
def getComponents(theta, b_azi):
    b_model_min = b_azi * math.cos(theta)
    b_model_max = b_azi * math.sin(theta)
    return b_model_min, b_model_max

    
def chiSquared2(RectArray, imp_Param): #No conversion is needed for this one
    RectArray = normalize(RectArray)
    numOfData = len(RectArray)
    chiSquare = 0    
    isEven = False
    radialDist = 0
    if len(RectArray) % 2 == 0:
        isEven = True    
    for x in range(numOfData):
        radialDist = getRadialDistance(imp_Param, numOfData, x, isEven)
        theta = getTheta(radialDist, imp_Param)
        b_axi, b_azi = modelFluxRope(radialDist)
        b_model_min, b_model_max = getComponents(theta, b_azi)
        chiSquare += ((RectArray[x][0] - b_model_max) ** 2) + ((RectArray[x][1] - b_axi) ** 2) + ((RectArray[x][2] - b_model_min)  ** 2)
    chiSquare = chiSquare / ((3 * numOfData) - 4)
    return chiSquare

def getRadialDistance(s, numOfDataPoints, index, isEven):
    totalLength = 2 * (math.sqrt(1 - (s**2)))
    distBtwn = totalLength / (numOfDataPoints - 1)
    distance = 0
    half = numOfDataPoints / 2
    if(isEven and (index == half) or (isEven and (index == (half + 1)))):
        distance = distBtwn * 0.5
    elif(isEven and (index > half)):
        distance = distBtwn * (((numOfDataPoints - 1) / 2) - index)
    elif(isEven and (index < half)):
        distance = distBtwn * (((numOfDataPoints - 1) / 2) - index)
    elif(index > (math.ceil(half))): #odd cases
        distance = (index - (math.ceil(half))) * distBtwn
    elif(index == (math.ceil(half))):
        distance = 0
    else:
        distance = (abs(index - half) * distBtwn)
    radialDist = math.sqrt((distance ** 2 ) + (s ** 2))
    return radialDist


    

        
    
        
    
    
    