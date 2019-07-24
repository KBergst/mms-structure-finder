# -*- coding: utf-8 -*-
"""
Created on Wed July 5 09:34:54 2019

@author: matthewchen37

Definitions for functions that are used for fitting 
"""


import math
from scipy import special as sp
from scipy import optimize as ot
import matplotlib.pyplot as plt2
import matplotlib.lines as mlines
import sympy as sy


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
        CylindArray[i][2] = theta #Stores angular direction in third position --> Azimuthal 
        CylindArray[i][1] = RectArray[i][1] #Z position left untouched, stored in 2rd position --> Axial Coordinate
     
    return CylindArray 

def getAzimuthal(maxVar, minVar):
    B_azi = math.sqrt((maxVar ^ 2) + (minVar ^ 2))
    return B_azi


def modelFluxRope(impact_param, a, b): #Mmodel flux rope based off of Smith et al., 2017
    #Elphic and Russell Non-Force Free Model
    B0 = 1 #Assume B0 is equal to 0
    #b = .35 Random, arbitrary constant number I thought fit well with the model 
    #a = .35 Another random arbitrary constant I thought would fit well with the model
    Br = B0 * math.exp( -1 * ((impact_param ** 2) / (b ** 2)))
    alpha = (math.pi / 2) * (1 - math.exp( -1 * ((impact_param ** 2) / (a ** 2))))
    B_axial = Br * math.cos(alpha)
    B_azimuthal = Br * math.sin(alpha)
    return B_axial, B_azimuthal

def modelForceFreeFluxRope(impact_param):
    #We assume that B0 and the Helicity are equal to 1 so that they are normalized
    B0 = 1 #should it equal 1? this is the missing piece?
    H = 1
    alpha = 2.4048 #From paper, constant alpha makes the flux rope linear
    B_axial = B0 * sp.jv(0, alpha * impact_param)
    B_azimuthal = B0 * H * sp.jv(1, alpha * impact_param)
    return B_axial,B_azimuthal # --> FORCE FREE MODEL'''




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
    

def chisquared1(RectArray, label): #first chi-squared test as defined by Smith et al., 2017
    CylindArray = RectToCylindrical(RectArray) 
    #not sure if this is correct -- made a lot of assumptions
    impactParameter = 0
    minChiSquared = 0
    chiSquaredValue = 0
    
    chiSquarePlotData = [0 for x in range(95)]
    chiSquareInput = [x for x in range(95)]
    
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
            chiSquaredValue += ((abs(CylindArray[x][0]) - B_azimuthal) ** 2) + ((abs(CylindArray[0][1]) - B_axial) ** 2)
            
            '''for  the model equation does it only give magnitudes in the cylindrical coordinate system?
               in the paper is the chi squared only comparing the magnitudes of the magnetic fields in the azimuthal direction?'''
        chiSquaredValue = chiSquaredValue / len(RectArray)
        
        chiSquarePlotData[y] = chiSquaredValue
        
        if imp == 0.0:
            minChiSquared = chiSquaredValue
        elif chiSquaredValue < minChiSquared:
            minChiSquared = chiSquaredValue
            impactParameter = imp 
            
    plotComponent(chiSquarePlotData, chiSquareInput, label)
    plotComponents(impactParameter, CylindArray, label)
    print("The impact parameter is" + str(impactParameter))
    
    
    
    if impactParameter > 0.5:
        
        minChiSquared = False
        return minChiSquared, impactParameter
    else:
        return minChiSquared, impactParameter 

def getTheta(radialDist, s):
    if (s == 0): #im not sure if this is correct
        theta = 0
    else:
        theta = math.acos(s / radialDist)
    return theta
        
    
def getComponents(theta, b_azi):
    b_model_min = b_azi * math.cos(theta)
    b_model_max = b_azi * math.sin(theta)
    b_model_min = abs(b_model_min)
    b_model_max = abs(b_model_max)
    return b_model_min, b_model_max

    
def chiSquared2(RectArray, imp_Param, label): #No conversion is needed for this one
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
    plotComponents3(imp_Param, RectArray, label)
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



def plotComponent(components1, xAxis, label):
    plt2.figure(1)
    plt2.plot(xAxis, components1, 'ro')
    plt2.title("Chi-Square Value over Impact Parameters 0 to 0.95")
    plt2.savefig('FittingPics/ImpactParameter' + str(label) + '.png')

def plotComponents(impactParam, array, label):
    
    isEven = False
    if len(array) % 2 == 0:
        isEven = True
  
    
    CylindArray = array
    plt2.figure(2)
    plt2.title("Model vs Data Azimuthal and Axial")
    for x in range(len(CylindArray)):
            radialDist = getRadialDistance(impactParam, len(array), x, isEven)
            B_axial, B_azimuthal = modelFluxRope(radialDist)
            plt2.plot(x, B_axial, marker='o', markersize=3, color="red")
            plt2.plot(x, B_azimuthal, marker='o', markersize=3, color="blue")
            plt2.plot(x, abs(CylindArray[x][0]), marker='o', markersize=3, color="green")
            plt2.plot(x, abs(CylindArray[x][1]), marker='o', markersize=3, color="yellow")
            
    
    
    b1 = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=10, label='Model B-Axial')
    b2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='Model B-Azimuthal')
    b4 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='Data B-Axial')
    b3 = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=10, label='Data B-Azimuthal')    
    
    
    plt2.legend(handles=[b1, b2, b4, b3])
    plt2.savefig('FittingPics/AzimuthalandAxial' + str(label) + '.png')
    plt2.show()
            
    
def plotComponents3(impactParam, array, label):
    numOfData = len(array)   
    isEven = False
    radialDist = 0
    plt2.figure(4)
    plt2.title("Chi-Squared 2")
    
    if len(array) % 2 == 0:
        isEven = True    
    for x in range(numOfData):
        radialDist = getRadialDistance(impactParam, numOfData, x, isEven)
        theta = getTheta(radialDist, impactParam)
        b_axi, b_azi = modelFluxRope(radialDist)
        b_model_min, b_model_max = getComponents(theta, b_azi) 
        plt2.plot(x, b_model_max, marker='o', markersize=3, color="red")
        plt2.plot(x, b_axi, marker='o', markersize=3, color="green")
        plt2.plot(x, b_model_min, marker='o', markersize=3, color="blue")
        plt2.plot(x, abs(array[x][0]), marker='o', markersize=3, color="magenta")
        plt2.plot(x, abs(array[x][1]), marker='o', markersize=3, color="cyan")
        plt2.plot(x, abs(array[x][2]), marker='o', markersize=3, color="yellow")
        
        
        
        
    b1 = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                          markersize=10, label='Model Max')
    b2 = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='Model Min')
    b3 = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=10, label='Model Axial')
    b4 = mlines.Line2D([], [], color='magenta', marker='o', linestyle='None',
                          markersize=10, label='Data Max')    
    b5 = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                          markersize=10, label='Data Min')
    b6 = mlines.Line2D([], [], color='cyan', marker='o', linestyle='None',
                          markersize=10, label='Data Axial') 
    
    plt2.legend(handles=[b1, b2, b3, b4, b5, b6])    
    plt2.savefig('FittingPics/Components' + str(label) + '.png')
    plt2.show()


def plotBvsR(DataArray, impactParam):        
    normArray = DataArray
    isEven = False
    if len(normArray) % 2 == 0:
        isEven = True  
        
    plt2.figure(5)
    plt2.title("Linearized Magnetic field and Radius Data(Supposedly)")
    magnitude = [0 for x in range(len(normArray))]
    for p in range(len(normArray)):
        magnitude[p] = math.log((normArray[p][0] ** 2) + (normArray[p][1] ** 2) + (normArray[p][2] ** 2))
        r = getRadialDistance(impactParam, len(normArray), p, isEven)
        r = r ** 2
        plt2.plot(r, magnitude[p], marker='o', markersize=3, color="green")
    plt2.show()
    


        
    
def derivativeBMax(s, a, b):
    B0 = 1
    x = sy.symbols('x', real=True)
    num1 = B0 * sy.exp(-(s**2 + x ** 2)/ b**2)
    num2 = sy.sin(sy.atan(sy.sqrt(s ** 2 + x ** 2) / s)) 
    num3 = sy.sin((math.pi/2) * (1 - sy.exp(-(s**2 + x ** 2)/ a**2)))
    bMax = num1 * num2 * num3
    bMaxPrime = sy.diff(bMax, x)
    print(str(bMaxPrime))
    zero1 = ot.brentq(sy.lambdify(x, bMaxPrime), -2, 0)
    zero2 = ot.brentq(sy.lambdify(x, bMaxPrime), 0, 2)
    return zero1, zero2 
    
    
def radiusRange(zeroesArray, s):
    lowerBound = math.sqrt((zeroesArray[0] ** 2) + (s ** 2))
    upperBound = math.sqrt((zeroesArray[0] ** 2) + (s ** 2))
    bounds = [lowerBound, upperBound]
    return bounds
    
    
    
    
    
    
    