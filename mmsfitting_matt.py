# -*- coding: utf-8 -*-
"""
Created on Wed July 5 09:34:54 2019

@author: matthewchen37

Definitions for functions that are used for fitting 
"""
import numpy as np
import cdflib #NEEDS python 3.7 to run
import math
import mmstimes as mt

#Turns (x,y,z) coordinates into (r, theta, z) coordinates 
def RectToCylindrical(RectArray):
    {
    row, column = len(RectArray), len(RectArray[0]) #Gets the Height and Width of Array for conversion
    CylindArray = [[0 for x in range(column)] for y in range[row]] #Initializes the return array as empty
    for i in range(row): #Each row is a 3 dimensional vector, this loops goes down each row
        x = RectArray[i][0]
        y = RectArray[i][1]
        r = math.sqrt((x ^ 2) + (y ^ 2)) # finds radial distance
        theta = math.atan(y/x) #Finds angle theta based off of x and y vectors, theta is in RADIANS
        CylindArray[i][0] = r #Store radial distance in first position --> Azimuthal
        CylindArray[i][1] = theta #Store theta in second position --> Azimuthal 
        CylindArray[i][2] = RectArray[i][2] #Z position left untouched, stored in 3rd position --> Axial Coordinate
     
    return CylindArray 
    }