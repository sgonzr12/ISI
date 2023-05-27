#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:58:23 2023

@author: mines46
"""

number = int(input("introduce un  numero"))

factorial = 1
    
for i in range(1, number+1):
    factorial *= i
        
print(f"el factoria: {number} es: {factorial}")