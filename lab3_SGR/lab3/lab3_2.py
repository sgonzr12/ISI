#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 16:19:58 2023

@author: mines46
"""

import numpy as np


matrix1 = np.random.rand(20, 20) * 3
print("matrix1 \n", matrix1)


print("\n indices entre 1 y 2 \n", np.nonzero(matrix1[(matrix1>1) & (matrix1<2)]))


print("\n indices  no entre 1 y 2 \n", np.nonzero(matrix1[(matrix1<1) | (matrix1>2)]))


round_matrix = np.round(matrix1)

print("\n round_matrix != 1 \n",  round_matrix[round_matrix != 1])