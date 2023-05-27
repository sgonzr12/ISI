#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 13:25:58 2022
Modified on Wed May 17 2023

@author: YOUR NAME HERE
"""

import pandas as pd
import os
import numpy as np
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
# -------------
# Program the PCA
# -------------
# Implementation of the PCA method, according to the algorithm studied during
# the theoretical lectures

    # For simplicity purposes, get the data into a numpy array
    iris_data = iris_df.copy().drop('variety', axis=1).to_numpy()

# 1. Data normalization: It will consist on standardizing the data (i.e.,
# subtracting to each variable the mean of the variable in all the samples of
# the dataset) and dividing it by the standard deviation of that variable.
# ====================== YOUR CODE HERE ======================

# Compute the mean and standard deviation of each column
        
means = np.mean(iris_data, axis=0)
stds = np.std(iris_data, axis=0)
        
# Subtract the means and divide by the standard deviations
        
iris_data = (iris_data - means) / stds

# ============================================================

# 2. Calculation of the covariance matrix
# ====================== YOUR CODE HERE ======================
covariance_matrix = np.cov(iris_data, rowvar=False)
# ============================================================

# 3. Getting eigenvalues and eigenvectors of the convariance matrix
# ====================== YOUR CODE HERE ======================
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
# ============================================================

# 4. Order of the eigenvectors according to the order of their corresponding
# associated eigenvalues, sorted in descenting order
# ====================== YOUR CODE HERE ======================
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]
# ============================================================

# 5. Selection of the first k eigenvectors
# In case k==0, the select a value of k so that 95% of the variance can be
# explained.
# Otherwise, select the first k eigenvectors
# ====================== YOUR CODE HERE ======================
if k == 0:
    # Select the minimum value of k that explains 95% of the variance
    total_variance = sum(eigenvalues)
    explained_variance = np.cumsum(eigenvalues) / total_variance
    k = np.argmax(explained_variance >= 0.95) + 1

# Select the first k eigenvectors
selected_eigenvectors = eigenvectors[:, :k]

# ============================================================

# 6. Getting the reduced dataset:
# It is the same as applying the rotation matrix to the data to obtain the
# values in the new coordinate system.
# ====================== YOUR CODE HERE ======================
iris_data_redux = np.dot(iris_data, selected_eigenvectors)
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
    ax_2D.set_ylabel("PC2")

    plt.title('Scatter plot of the reduced data', fontsize=14)
    fig_2D.show()
