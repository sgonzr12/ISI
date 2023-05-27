#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:19:51 2022
Modified on March 2023

@author: YOUR NAME HERE
"""

import h5py
import os
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn import preprocessing
from sklearn import svm

# %%
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    np.random.seed(323)

    dir_data = "Data"
    data_path = os.path.join(dir_data, "mammographic_data.h5")


# %%
# -------------
# PRELIMINARY: LOAD DATASET
# -------------

    # import features and labels
    h5f_data = h5py.File(data_path, 'r')

    features_ds = h5f_data['data']
    labels_ds = h5f_data['labels']

    X = np.array(features_ds)
    y = np.array(labels_ds)
    y = np.reshape(y, newshape=(y.shape[0],))

    h5f_data.close()

# %%
# -------------
# PART 1: CREATE K FOLDS AND CHECK THE PROPORTIONS
# -------------
    K = 10  # number of folds

    # K-Folds cross-validator instantiator
    # ====================== YOUR CODE HERE ======================
    kfold = KFold(n_splits=K, shuffle=True)
    # ============================================================

    proportion_class_0 = np.sum(y == 0) / y.size
    proportion_class_1 = 1 - proportion_class_0
    print("**********************************************************")
    print("*********** CLASS PROPORTIONS WITHIN THE FOLDS ***********")
    print("**********************************************************")
    print("\n")
    print("The distribution of the complete dataset is:")
    print("  - {:.2f} % elements of class 0".format(
        100 * proportion_class_0))
    print("  - {:.2f} % elements of class 1".format(
        100 * proportion_class_1))
    print("\n")
    print("The distribution of the elements within each fold is:")

    # ====================== YOUR CODE HERE ======================
    # This goes within a for loop. Write the for here
    for i, (train_index, test_index) in enumerate(kfold.split(X)):
        proportion_class_0_train = np.sum(
            y[train_index] == 0) / train_index.size
        proportion_class_1_train = 1 - proportion_class_0_train
        proportion_class_0_test = np.sum(
            y[test_index] == 0) / test_index.size
        proportion_class_1_test = 1 - proportion_class_0_test
        print("* FOLD {}:".format(i+1))
        print("  - TRAIN: {:.2f} % elements of class 0;     {:.2f} % elements of class 1".format(
              100 * proportion_class_0_train, 100 * proportion_class_1_train))
        print("  - TEST: {:.2f} % elements of class 0;      {:.2f} % elements of class 1".format(
              100 * proportion_class_0_test, 100 * proportion_class_1_test))
        print("\n")
    # ============================================================


    # Stratified K-Folds cross-validator instantiator
    # ====================== YOUR CODE HERE ======================
    skf = StratifiedKFold(n_splits=K, shuffle=True)
    # ============================================================

    proportion_class_0 = np.sum(y == 0) / y.size
    proportion_class_1 = 1 - proportion_class_0
    print("***********************************************************")
    print("***** CLASS PROPORTIONS WITHIN THE FOLDS (STRATIFIED) *****")
    print("***********************************************************")
    print("\n")
    print("The distribution of the elements within each fold is:")

    # ====================== YOUR CODE HERE ======================
    # This goes within a for loop. Write the for here
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
        proportion_class_0_train = np.sum(
            y[train_index] == 0) / train_index.size
        proportion_class_1_train = 1 - proportion_class_0_train
        proportion_class_0_test = np.sum(
            y[test_index] == 0) / test_index.size
        proportion_class_1_test = 1 - proportion_class_0_test
        print("* FOLD {}:".format(i+1))
        print("  - TRAIN: {:.2f} % elements of class 0;     {:.2f} % elements of class 1".format(
              100 * proportion_class_0_train, 100 * proportion_class_1_train))
        print("  - TEST: {:.2f} % elements of class 0;      {:.2f} % elements of class 1".format(
              100 * proportion_class_0_test, 100 * proportion_class_1_test))
        print("\n")
    # ============================================================
        
# %%
# -------------
# PART 2: CROSS VALIDATION WITH SVM
# -------------

    # Parameters for SVM
    C_value = 1
    kernel_type = "linear"
    
    # Initialization of the vectors to store the accuracies and Fscores
    # of each fold
    i = 0
    accuracies = np.zeros(shape=(K,))
    Fscores = np.zeros(shape=(K,))

    # For loop to obtain the accuracy and fscore of the SVM classifier at each
    # fold
    # Try using the stratified and non-stratified k-fold cross validator, to
    # see if there is any difference
    # ====================== YOUR CODE HERE ======================
    # This goes within a for loop. Write the for here
    for train_index, test_index in skf.split(X, y):
 
        # Extract train and test subsets of this fold
        # ====================== YOUR CODE HERE ======================
        X_train_fold = X[train_index]
        y_train_fold = y[train_index]
        X_test_fold = X[test_index]
        y_test_fold = y[test_index]
        # ============================================================

        # Standardize data of this fold
        scaler = preprocessing.StandardScaler()
        scaler.fit(X_train_fold)
        X_train_fold = scaler.transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)

        # Train the SVM and classify with the corresponding subsets
        # Instantiate the SVM with the defined kernel type and C value, train
        # it and use it to classify. Use the train and test sets of the current
        # iteration. 
        # ====================== YOUR CODE HERE ======================
        clf = svm.SVC(kernel=kernel_type)
        
        # Train
        clf.fit(X_train_fold, y_train_fold)
        
        # Classify test set
        y_test_assig_fold =  clf.predict(X_test_fold)
        # ============================================================

        # Compute performance metrics of the test set in this fold and store
        # them
        # ====================== YOUR CODE HERE ======================
        tp = np.sum((y_test_fold == 1) & (y_test_assig_fold == 1))
        tn = np.sum((y_test_fold == 0) & (y_test_assig_fold == 0))
        fn = np.sum((y_test_fold == 0) & (y_test_assig_fold == 1))
        fp = np.sum((y_test_fold == 1) & (y_test_assig_fold == 0))
        
        accuracy_fold = (tp+tn)/(tp+tn+fp+fn)
        
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        Fscore_fold = (precision * recall)/(precision+recall)
        
        accuracies[i] = accuracy_fold
        Fscores[i] = Fscore_fold
        # ============================================================

        i += 1
    # ============================================================

# %%
# -------------
# PART 3: SHOW FINAL RESULTS
# -------------

    print("\n\n")
    print('***********************************************')
    print('******* RESULTS OF THE CROSS VALIDATION *******')
    print('***********************************************')
    print('\n')

    for i in range(K):
        print("FOLD {}:".format(i+1))
        print("    Accuracy = {:4.3f}".format(accuracies[i]))
        print("    Fscore = {:5.3f}".format(Fscores[i]))

    # ====================== YOUR CODE HERE ======================
    # Calculate mean and std of the accuracies and F1-scores
    mean_accuracy = sum(accuracies)/K
    std_accuracy = np.std(accuracies)
    mean_fscore = sum(Fscores/K)
    std_fscore = np.std(Fscores)
    # ============================================================
    
    print("\n")
    print("AVERAGE ACCURACY = {:4.3f}; STD ACCURACY = {:4.3f}".format(
        mean_accuracy, std_accuracy))
    print("AVERAGE FSCORE = {:4.3f}; STD FSCORE = {:4.3f}".format(
        mean_fscore, std_fscore))
    print("\n")
    print('***********************************************')
    print('***********************************************')
    print('***********************************************')
    print("\n\n")

# %%
# -------------
# PART 4: CROSS-VALIDATION ESTIMATES WITH SCIKIT-LEARN
# -------------

    # ====================== YOUR CODE HERE ======================
    # Create a new instance of the SVM classifier with linear kernel
    svm = svm.SVC(kernel='linear', random_state=42)


    # Carry out the cross validation with the function cross_validate
    scoring = ['accuracy', 'f1_macro']
    scores = cross_validate(svm, X, y, scoring=scoring, cv=K)
    # ============================================================
    
    
    print("\n\n")
    print('********************************************************')
    print('******* RESULTS OF THE CROSS VALIDATION (SIMPLE) *******')
    print('********************************************************')
    print('\n')
    
    # Get the accuracies and fscores from the output of cross_validate
    # ====================== YOUR CODE HERE ======================
    accuracies_skl = scores['test_accuracy']
    fscores_skl = scores['test_f1_macro']

    mean_accuracy_skl = np.mean(accuracies_skl)
    std_accuracy_skl = np.std(accuracies_skl)
    mean_fscore_skl = np.mean(fscores_skl)
    std_fscore_skl = np.std(fscores_skl)
    # ============================================================

    for i in range(K):
        print("FOLD {}:".format(i+1))
        print("    Accuracy = {:4.3f}".format(accuracies_skl[i]))
        print("    Fscore = {:5.3f}".format(fscores_skl[i]))

    print("\n")
    print("AVERAGE ACCURACY = {:4.3f}; STD ACCURACY = {:4.3f}".format(
        mean_accuracy_skl, std_accuracy_skl))
    print("AVERAGE FSCORE = {:4.3f}; STD FSCORE = {:4.3f}".format(
        mean_fscore_skl, std_fscore_skl))
    print("\n")
    print('***********************************************')
    print('***********************************************')
    print('***********************************************')
    print("\n\n")
