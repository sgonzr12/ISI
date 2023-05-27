#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 11:16:39 2023

@author: mines46
"""

import random
import math

fscores = open("scores.txt", "r")
scores = fscores.readlines()
fscores.close()

points = {}

for line in scores:
   readname, readpoints = line.strip().split(':')
   points[readname] = readpoints

mostrarscore = int(input("pulsa 1 si quieres ver las puntuaciones\n"))

if mostrarscore == 1:
    for user, point in points.items():
        print(f"{user}:{point}")

continuar = True
while(continuar):

    nombre = input("introduce tu nombre\n")
    N = int(input("introduce el número máximo\n"))
    
    try:
        print (f"tu maxima puntuación es: {points[nombre]}")
    except:
        points[nombre] = 0
    
    numero_aleatorio = random.randint(1, N)
    tryals = int(math.log2(N))
    
    point = 0
    
    for i in range(1, tryals+1):
        
        prueba = int(input(f"intenta adivinar el número entre 1 y {N}\n"))
        
        if prueba == numero_aleatorio:
            
            point = N/(2*i-1)
            
            break
        else:
        
            print(f"{tryals-i} intentos restantes") 
            
            if prueba < numero_aleatorio:
                print(f"el número es mayor que {prueba}")
            else:
                print(f"el número es menor que {prueba}")
    
    print(f"has conseguido {point} puntos")
    if points[nombre]<point:
        points[nombre] = point
    
    fpuntuacion = open("scores.txt", "w")
    
    for user, point in points.items() :
        fpuntuacion.write(f"{user}:{point}")
    
    if input("pulsa 1 para volver a jugar\n") != 1:
        continuar = False
