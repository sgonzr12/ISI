#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:20:17 2023

@author: mines46
"""

dni = int(input("introduce un dni\n"))
resto = dni % 23
tabla = ('T','R','W','A','G','M','Y','F','P','D','X','B','N','J','Z','S','Q','V','H','L','C','K','E')
print(tabla[resto])