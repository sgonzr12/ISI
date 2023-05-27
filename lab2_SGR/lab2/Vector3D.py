#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:14:31 2023

@author: mines46
"""

import math
import pickle

class Vector3D:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def changeValue(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def printVector(self):
        print(f"las coordenadas del vector son: x = {self.x}, y = {self.y}, z = {self.z}")
    
    def addVector(self, vector):
        self.x += vector.x
        self.y += vector.y
        self.z += vector.z
        
    def subVector(self, vector):
        self.x -= vector.x
        self.y -= vector.y
        self.z -= vector.z
        
    def scalarmult(self, a):
        self.x *= a
        self.y *= a
        self.z *= a
    
    def modulus(self):
        return math.sqrt(math.pow(self.x,2)+math.pow(self.y,2)+math.pow(self.z,2))
    
    def storetxt(self):
        
        fdestino = open("vector.txt", 'w')
        
        fdestino.write(f"las coordenadas del vector son: x = {self.x}, y = {self.y}, z = {self.z}")
        
        fdestino.close()
        
        
    def storepickle(self):
        
        fdestino = open("vector.pkl", 'wb')
        
        pickle.dump(self, fdestino)
        
        fdestino.close()
        