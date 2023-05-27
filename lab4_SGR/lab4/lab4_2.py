#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 13:01:41 2023

@author: mines46
"""

import matplotlib.pyplot as plt
import numpy as np

parNum = int(input("Insert the number of particles: "))

centerX = 0.0
centerY = 0.0
total_mass = 0.0

partsX = np.zeros(parNum)
partsY = np.zeros(parNum)
partsMass = np.zeros(parNum)

for i in range(parNum):
    partsX[i] = float(input("Particle {}. Position x: ".format(i+1)))
    partsY[i] = float(input("Particle {}. Position y: ".format(i+1)))
    partsMass[i] = float(input("Particle {}. Mass: ".format(i+1)))
    
    total_mass += partsMass[i]
    centerX += partsMass[i] * partsX[i]
    centerY += partsMass[i] * partsY[i]
    
centerX /= total_mass
centerY /= total_mass
    
plt.close('all')
fig = plt.figure(figsize=(7, 7))

plt.title("center of mass") 
plt.grid(True)
plt.scatter(partsX, partsY, s = partsMass, c = "b")
plt.scatter(centerX, centerY, s = total_mass, c = "g", marker = "^")

for i in range(parNum):
    plt.annotate(partsMass[i], (partsX[i], partsY[i]))

plt.annotate(total_mass,(centerX,centerY))