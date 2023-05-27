#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:11:33 2023

@author: mines46
"""

from Vector3D import Vector3D

vector = Vector3D(0, 0, 0)
vector.printVector()

vector.changeValue(-6, 10, 5)
vector.printVector()

vector2 = Vector3D(5, -1, 0)
vector.printVector()

vector.addVector(vector2)
vector.printVector()

vector2 = Vector3D(-1, -1, -9)
vector.printVector()

vector.subVector(vector2)
vector.printVector()

vector.scalarmult(3.5)
vector.printVector()

vector.modulus()
vector.printVector()

vector.storetxt()

vector.storepickle()