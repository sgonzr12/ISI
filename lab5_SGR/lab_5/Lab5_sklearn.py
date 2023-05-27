6#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 16:12:41 2022
Modified on Mon Mar 13 2023

@author: CHANGE THE NAME

This script carries out a classification experiment of the spambase dataset by
means of the kNN classifier, USING THE SCIKIT-LEARN PACKAGE
"""

# Import whatever else you need to import
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# %%
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    # %% PART 1: LOAD DATASET AND TRAIN-TEST PARTITION

    # Load csv with data into a pandas dataframe
    """
    Each row in this dataframe contains a feature vector, which represents an
    email.
    Each column represents a variable, EXCEPT LAST COLUMN, which represents
    the true class of the corresponding element (i.e. row): 1 means "spam",
    and 0 means "not spam"
    """
    dir_data = "Data"
    spam_df = pd.read_csv(os.path.join(dir_data, "spambase_data.csv"))
    y_df = spam_df[['Class']].copy()
    X_df = spam_df.copy()
    X_df = X_df.drop('Class', axis=1)

    # Convert dataframe to numpy array
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    """
    Parameter that indicates the proportion of elements that the test set will
    have
    """
    proportion_test = 0.3

    """
    Partition of the dataset into training and test sets is done. 
    Use the function train_test_split from scikit_learn
    """
    # ====================== YOUR CODE HERE ======================
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    # ============================================================

    """
    Create an instance of the kNN classifier using scikit-learn
    """
    # ====================== YOUR CODE HERE ======================
    knn = KNeighborsClassifier(n_neighbors=5)
    # ============================================================

    """
    Train the classifier
    """
    # ====================== YOUR CODE HERE ======================
    knn.fit(x_train, y_train.ravel())
    # ============================================================

    """
    Get the predictions for the test set samples given by the classifier
    """
    # ====================== YOUR CODE HERE ======================
    y_pred = knn.predict(x_test)
    # ============================================================
    
    """
    Show the confusion matrix. Use the same methods that were used in the
    first part of the lab (i.e., see Lab5.py)
    """
    # ====================== YOUR CODE HERE ======================
    confusion_matrix_kNN = confusion_matrix(y_true=y_test, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_kNN)
    disp.plot()
    plt.title('Confusion Matrix', fontsize=14)
    plt.show()
    # ============================================================
