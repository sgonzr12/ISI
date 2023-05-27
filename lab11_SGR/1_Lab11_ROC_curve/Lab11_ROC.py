#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:31:33 2022
Modified on Wed May 17 2023

@author: YOUR NAME HERE
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def get_ROC(y_test, y_test_hat, decision_thresholds):
    """
    This function obtains the values of the 1-specificity (False Positive Rate
     - FPR) and the senstivity (True Positive Rate - TPR) that define the ROC
    curve, for some decision thresholds. Besides, it computes the area under
    the calculated ROC curve. This function is only designed for binary
    classification problems.

    Parameters
    ----------
    y_test : Numpy vector
        Vector that contains the real classes (0 or 1) of the samples in the
        test set.
    y_test_hat : Numpy vector
        Vector that contains the probanilities of belonging to class 1 yielded
        by the classifier, for each sample in the test set.
    decision_thresholds : Numpy vector
        The different decision thresholds that will be used to calculate the
        sensitivity and 1-specificity values.

    Returns
    -------
    v_FPR : Numpy Vector
        Vector that contains the values of 1-specificity (i.e. FPR) for each
        decision threshold.
    v_FPR : Numpy Vector
        Vector that contains the values of sensitivity (i.e. TPR) for each
        decision threshold.
    """

    v_FPR = []
    v_TPR = []

    # ====================== YOUR CODE HERE ======================
    
    for threshold in decision_thresholds:
        predicted_labels = np.where(y_test_hat >= threshold, 1, 0)

        TP = np.sum((y_test == 1) & (predicted_labels == 1))
        TN = np.sum((y_test == 0) & (predicted_labels == 0))
        FP = np.sum((y_test == 0) & (predicted_labels == 1))
        FN = np.sum((y_test == 1) & (predicted_labels == 0))

        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)

        v_FPR.append(FPR)
        v_TPR.append(TPR)
        
    v_FPR = np.array(v_FPR)
    v_TPR = np.array(v_TPR)

    # ============================================================

    return v_FPR, v_TPR


# %%
# -------------
# MAIN PROGRAM
# -------------
if __name__ == "__main__":

    np.random.seed(523)

    dir_data = "Data"
    data_path = os.path.join('..', dir_data, "mammographic_data.csv")
    test_size = 0.3

# -------------
# PRELIMINARY: LOAD DATASET AND PARTITION TRAIN-TEST SETS (NO NEED TO CHANGE
# ANYTHING)
# -------------

    # import features and labels
    # Load the data
    data_df = pd.read_csv(data_path)
    y = data_df['Class'].to_numpy()
    X = data_df.copy().drop('Class', axis=1).to_numpy()

    # SPLIT DATA INTO TRAINING AND TEST SETS
    (X_train, X_test, y_train, y_test) = train_test_split(X, y,
                                                          test_size=test_size,
                                                          random_state=None)

    # NORMALIZE DATA (normalization by standardization of the features)
    # ====================== YOUR CODE HERE ======================

    # Compute the mean and standard deviation of each column
    
    means = np.mean(X_train, axis=0)
    stds = np.std(X_train, axis=0)
    
    # Subtract the means and divide by the standard deviations
    
    X_train = (X_train - means) / stds
    X_test = (X_test - means) / stds
    
    # ============================================================

    # Uncomment if you want to check that the mean and std are close to 0 and 1
    # respectively
    #print("Mean of the training set: {}".format(X_train.mean(axis=0)))
    #print("Std of the training set: {}".format(X_train.std(axis=0)))
    #print("Mean of the test set: {}".format(X_test.mean(axis=0)))
    #print("Std of the test set: {}".format(X_test.std(axis=0)))

# %%
# -------------
# PART 1: CLASSIFICATION WITH SCIKIT-LEARN'S LOGISTIC REGRESSION
# -------------

    # Instance of the logistic regression model
    logit_model = LogisticRegression()

    # Train the model
    logit_model.fit(X_train, y_train.reshape(y_train.shape[0],))

    # Predict the probabilities of belonging to class 1 of the test samples
    y_test_hat = logit_model.predict_proba(X_test)[:, 1]

    # Predict the classes of the test set samples
    y_test_assig = logit_model.predict(X_test)

    # Test of the probabilities. If everything is right, it should print True
    # y_test_assig_proba = y_test_hat >= 0.5
    # print((y_test_assig == y_test_assig_proba).all())

    # Display confusion matrix when the decision threshold is 0.5
    confm = confusion_matrix(y_true=y_test, y_pred=y_test_assig)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm)
    disp.plot()
    plt.title("Confusion Matrix for the logistic regression classifier",
              fontsize=14)
    plt.show()

# %%
# -------------
# PART 2: COMPUTATION AND PLOT OF THE ROC CURVE
# -------------

    # Initialization of the decision thresholds
    # Give values to the decision thresholds that will be used in the
    # calculation of the ROC curve
    # ====================== YOUR CODE HERE ======================
    decision_thresholds = np.arange(0, 1.01, 0.01)
    # ============================================================

    y_test = np.reshape(y_test, (y_test.shape[0],))

    # Calling the function to bould the ROC curve
    v_FPR, v_TPR = get_ROC(y_test, y_test_hat, decision_thresholds)

    # AREA UNDER THE ROC CURVE
    #    Integration of the ROC curve usint the trapezoidal rule
    AUC = np.trapz(y=np.flip(v_TPR), x=np.flip(v_FPR))

    # Plot of the curve
    plt.figure(2)
    plt.plot(v_FPR, v_TPR, 'b-', label="ROC of classifier")
    plt.plot([0, 1], [0, 1], 'r--', label="Random classification")
    plt.legend(loc='lower right', shadow=True)
    plt.xlabel("FPR (1-specificity)")
    plt.ylabel("TPR (sensitivity)")
    plt.title("ROC curve (AUC={:.3f})".format(AUC), fontsize=14)
    plt.xlim([-0.001, 1.001])
    plt.ylim([-0.001, 1.001])
    plt.show()
