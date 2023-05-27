#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 20:00:57 2023

@author: mines46
"""

from urllib.request import urlopen
    
try:
    
    forigen = urlopen("https://www.gutenberg.org/cache/epub/1184/pg1184.txt")

    texto = forigen.read()

    palabras = len(texto.split())

    print(f"la url tiene {palabras} palabras")
    
except:
    
    print("la url no existe")
    



