#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 10:55:17 2023

@author: mines46
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas

iris_df = pandas.read_csv("iris.csv")
sepal_length_df = iris_df["sepal_length"]
sepal_width_df = iris_df["sepal_width"]
petal_length_df = iris_df["petal_length"]
petal_width_df = iris_df["petal_width"]
variety_df = iris_df["variety"]

colors = []
labels = []

plt.close('all')

fig = plt.figure(figsize = (7, 7))
ax = plt.axes(projection = "3d")

seto = 0
vers = 0
vir = 0

i = 0

for row in variety_df:
  
    if row == "Setosa":
        colors.append("b")
        seto = i
        labels.append("Setosa")
    elif row == "Versicolor":
        colors.append("r")
        vers = i
        labels.append("Versicolor")
    elif row == "Virginica":
        colors.append("g")
        vir = i
        labels.append("Virginica")
    i+=1
ax.scatter(sepal_length_df, sepal_width_df, petal_length_df, s = petal_width_df, c = colors)

ax.text(sepal_length_df[seto],sepal_width_df[seto], petal_length_df[seto], "Setosa",)
ax.text(sepal_length_df[vers],sepal_width_df[vers], petal_length_df[vers], "Versicolor",)
ax.text(sepal_length_df[vir],sepal_width_df[vir], petal_length_df[vir], "Virginica",)


plt.title("scatter plot of the iris dataset")
ax.set_xlabel("sepal_length")
ax.set_ylabel("sepal_width")
ax.set_zlabel("petal_length")

ax.scatter(5.1,3.5,1.4, s= 0.2, c = "r")
ax.scatter(5.1,3.5,1.4, s= 0.2, c = "g")

ax.legend(labels = ['Setosa', 'Versicolor', 'Virginica'])

plt.show()