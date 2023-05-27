#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 12:02:35 2023

@author: mines46
"""

sentence = input("introduce una frase\n")

print("1) convert the sentence into uppercase\n")
print("2) convert the sentence into lowercase\n")
print("3) convert the first character of each word into uppercase\n")
print("4) convert the characters that are in even positions into uppercase")

option = int(input("elige una opcion\n"))

if option == 1:
    uppersentence = sentence.upper()
    print(uppersentence)
elif option == 2:
    lowersentence = sentence.lower()
    print(lowersentence)
elif option == 3:
    title = sentence.title()
    print(title)
else:
    charlist = list(sentence)
    for i in range(1, len(charlist), 2):
        charlist[i] = charlist[i].upper()
    otherString = "".join(charlist)
    print(otherString)