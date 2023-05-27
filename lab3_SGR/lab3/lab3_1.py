# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

Matrix1 = np.array([[4, -2, 7], 
                    [9, 4, 1], 
                    [5, -1, 5]])
print("matrix1: \n", Matrix1)


Matrix2 = np.transpose(Matrix1)
print("\n matrix2 \n", Matrix2)


print("\n matrix1*matrix2 \n", Matrix1*Matrix2)


prodM1M2 = Matrix1@Matrix2
print("\n matrix1@matrix2 \n", prodM1M2)


prodM2M1 = Matrix2@Matrix1
print("\n matrix2@matrix1 \n", prodM2M1)


mat_corners = np.array([[Matrix1[0,0], Matrix1[0,-1]], 
                        [Matrix1[-1,0], Matrix1[-1,-1]]])

print("\n mat_corners \n", mat_corners)


vec_max = np.max(Matrix1, axis=1)
print("\n vec_max \n", vec_max)
print("\n glb max \n", np.max(Matrix1))


vec_min = np.min(Matrix1, axis=0)
print("\n vec_min \n", vec_min)
print("\n glb min \n", np.min(Matrix1))


vec_max = vec_max.reshape(1, -1)
vec_min = vec_min.reshape(-1, 1)
print("\n vec_min@vec_max \n", vec_min@vec_max)

mat_sum = np.sum(Matrix1[:,[0,2]])
print("|n mat_sum \n", mat_sum)