#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:28:14 2023

@author: mines46
"""
finalizar = True
numbers =[]

while finalizar:
    
    try:
        number = int(input("introduce un numero"))
    except ValueError:
        print("you might insert numbers")
    
    if number > 0:
        numbers.append(number)
    else:
        numbers.append(number)
        finalizar = False

squares =[]

print("los numeros de la lista son: ")

for i in numbers:
    print (i)

print("los cuadrados de los numeros introducidos son: " )
for i in numbers:
    square = pow(i,2)
    squares.append(square)
    print(square)

sum = 0

for i in squares:
    sum = sum + i

print(f"la suma de los cuadrados es: {sum}")