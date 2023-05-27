#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:13:40 2023

@author: mines46
"""

import math

def addValues(args):
    
    suma = 0
    for i in args:
        suma += i
    
    return suma
    
def subtract(*args):
    
    resta = args[0]
    resta -= args[1]
    
    return resta

def mult(args):
    
    suma = 1
    for i in args:
        suma *= i
    
    return suma

def div(*args):
    
    div = args[0]/args[1]
    
    return div

def potencia(*args):
    
    potencia = args[0]**args[1]
    
    return potencia

def log(*args):
    
    log = math.log(args[0])
    
    return log


def imprimeMenu():
    
    print("0)salir\n")
    print("1)Add an arbitrary number of values\n")
    print("2)Subtract two values\n")
    print("3)Multiply an arbitrary number of values\n")
    print("4)Divide two values\n")
    print("5)Calculate the value of one number raised to another\n")
    print("6)Calculate the natural logarithm of a number\n")

def main():
    imprimeMenu()
    option = int(input("elige una opción\n"))
    
    while option != 0: 
        if option == 1:
            
            entrada = input("introduce los números separados por espacio:\n ")
            lista = entrada.split(" ");
            lista2 = []
            for i in lista:
                lista2.append(int(i)) 
            
            print("el resultado de la suma es: ", addValues(lista2))
            
        elif option == 2:
            
            num1 = int(input("introduce el primer número"))
            num2 = int(input("introduce el segundo número"))
    
            print("el resultado de la resta es: ", subtract(num1, num2))
            
        elif option == 3:
            
            entrada = input("introduce los números separados por espacio:\n ")
            lista = entrada.split(" ");
            lista2 = []
            for i in lista:
                lista2.append(int(i)) 
            
            print("el resultado de la multiplicación es: ", mult(lista2))
            
        elif option == 4:
            
            num1 = int(input("introduce el primer número"))
            num2 = int(input("introduce el segundo número"))
    
            print("el resultado de la resta es: ", div(num1, num2))
            
        elif option == 5:
            
            num1 = int(input("introduce el primer número"))
            num2 = int(input("introduce el segundo número"))
    
            print("el resultado de la resta es: ", potencia(num1, num2))
            
        elif option == 6:
            
            num1 = int(input("introduce el número"))
    
            print("el resultado de la resta es: ", log(num1))
            
            
            
        imprimeMenu()
        option = int(input("elige una opción\n"))

if __name__ == "__main__":
    main()