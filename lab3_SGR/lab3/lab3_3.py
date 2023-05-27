#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:39:57 2023

@author: mines46
"""

import numpy as np

matrix = np.random.rand(10, 4) * 20 - 10
print("matrix \n", matrix)


euclideanMatrix = np.sqrt(np.sum((matrix[:, np.newaxis, :] - matrix) ** 2, axis=2))

for i in range(10):
    for j in range(i+1, 10):
        dist = euclideanMatrix[i,j]
        if dist < 10:
            print(f"The Euclidean distance between vectors {i} and {j} is {dist}")