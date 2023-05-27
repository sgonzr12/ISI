#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:17:23 2022
Modified on Wed May 17 2023

@author: YOUR NAME HERE
"""

import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# %%
# -------------
# MAIN PROGRAM
# -------------
"""
This is a script that carries out a PCA over some input data. This script reads
the iris dataset from an input CSV file and carries out a PCA, where the
selection of the number of variables will be carried out manually, i.e.
indicating the number of variables to take, or automatically, i.e. measuring
the amount of variance we want the reduced data to explain
"""
if __name__ == "__main__":

    # Number of target principal components
    k = 2

    # Load the data
    dir_data = os.path.join("..", "Data")
    iris_df = pd.read_csv(os.path.join(dir_data, "iris.csv"))


# %%
# -------------
# Plot the original data
# -------------
# A scatter plot in 3D is created. The first three variables of the data
# (i.e., sepal length, sepal width and petal length) indicate the position,
# whereas the fourth one (i.e. petal width) indicates the size of the point.
# The class of the sample determines the color of the corresponding point

    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

    iris_class = iris_df['variety'].to_list()
    names_classes = list(set(iris_class))

    colors_flowers = {names_classes[0]: "darkblue",
                      names_classes[1]: "turquoise",
                      names_classes[2]: "tomato"}

    fig_3D = plt.figure(num=1, figsize=(8, 8))
    ax_3D = fig_3D.add_subplot(projection='3d')
    mean_variety_df = iris_df.groupby('variety').mean()
    for label in names_classes:
        indices_label = iris_df['variety'] == label

        ax_3D.scatter(iris_df.loc[indices_label, 'sepal_length'],
                      iris_df.loc[indices_label, 'sepal_width'],
                      iris_df.loc[indices_label, 'petal_length'],
                      s=iris_df.loc[indices_label, 'petal_width'] * 20,
                      c=colors_flowers[label],
                      marker='o')

        coordinates_label = mean_variety_df.loc[label, :].to_numpy()
        ax_3D.text(coordinates_label[0], coordinates_label[1],
                   coordinates_label[2], label)

    ax_3D.legend(names_classes)

    ax_3D.set_xlabel(iris_df.columns[0])
    ax_3D.set_ylabel(iris_df.columns[1])
    ax_3D.set_zlabel(iris_df.columns[2])

    plt.title('Scatter plot of the original data', fontsize=14)
    fig_3D.show()

# %%
# --------------------------
# PCA with scikit-learrn
# --------------------------

    # For simplicity purposes, get the data into a numpy array
    iris_data = iris_df.copy().drop('variety', axis=1).to_numpy()

    # ====================== YOUR CODE HERE ======================
    pca = PCA(n_components = k)

    # Fit the PCA model to the data
    pca.fit(iris_data)

    # Get the transformed data in the new coordinate system
    iris_data_redux = pca.transform(iris_data)
    # ============================================================

# %%
# -------------
# Plot the reduced data
# -------------
# If k==2, a 2D scatter plot is created. The two resulting variables indicate
# the position, whereas the radius of the points is determined by the petal
# width of the original data
# The class of the sample determines the color of the corresponding point

if k == 2:
    fig_2D = plt.figure(num=2, figsize=(7, 7))
    ax_2D = fig_2D.add_subplot()
    for label in names_classes:
        indices_label = iris_df['variety'] == label
        ax_2D.scatter(x=iris_data_redux[indices_label, 0],
                      y=iris_data_redux[indices_label, 1],
                      s=iris_df.loc[indices_label, 'petal_width'] * 20,
                      c=colors_flowers[label],
                      marker='o')

        coordinates_label_2D = np.mean(iris_data_redux[indices_label, :],
                                       axis=0)
        ax_2D.text(x=coordinates_label_2D[0],
                   y=coordinates_label_2D[1],
                   s=label)
    ax_2D.legend(names_classes)
    ax_2D.set_xlabel("PC1")
    ax_2D.set_xlabel("PC2")

    plt.title('Scatter plot of the reduced data', fontsize=14)
    fig_2D.show()
