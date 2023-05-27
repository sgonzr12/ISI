6#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 17:28:45 2023

@author: mines46
"""

# Import whatever else you need to import
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
import numpy as np


# %%
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    # %% PART 1: LOAD DATASET AND TRAIN-TEST PARTITION

    # Load csv with data into a pandas dataframe

    dir_data = "Data"
    spam_df = pd.read_csv(os.path.join(dir_data, "data_inserts_Medium-high.csv"))
    y_df = spam_df[['Class']].copy()
    X_df = spam_df.copy()
    X_df = X_df.drop('Class', axis=1)

    # Convert dataframe to numpy array
    X = X_df.to_numpy()
    y = y_df.to_numpy()
    
    """
    k-Fold statified instantiator
    """
    K = 10;
        
    """
    Create an instance of the kNN classifier using scikit-learn
    """

    knn = KNeighborsClassifier(n_neighbors=5, metric = "manhattan")
    #metrics(manhattan, euclidean)
    
    scoring = ['accuracy', 'f1_macro', 'precision', 'recall']
    scores = cross_validate(knn, X, y, scoring=scoring, cv=K)
   
    #accuracy
    print("***************")
    print("The accuracy of the KNN classifier is {:.4f}".
          format(np.mean(scores['test_accuracy'])))
    print("***************")

    #precision
    print("")
    print("***************")
    print("The precision of the KNN classifier is {:.4f}".
          format(np.mean(scores['test_precision'])))
    print("***************")
    
    #recall
    print("")
    print("***************")
    print("The recall of the KNN classifier is {:.4f}".
          format(np.mean(scores['test_recall'])))
    print("***************")
    
    # F1 score
    print("")
    print("***************")
    print("The F1-score of the KNN classifier is {:.4f}".
          format(np.mean(scores['test_f1_macro'])))
    print("***************")
    