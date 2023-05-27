"""
Created on Wed Nov 16 13:31:33 2022

@author: Víctor González Castro
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # ============================================================

    # Uncomment if you want to check that the mean and std are close to 0 and 1
    # respectively
    # print("Mean of the training set: {}".format(X_train.mean(axis=0)))
    # print("Std of the training set: {}".format(X_train.std(axis=0)))
    # print("Mean of the test set: {}".format(X_test.mean(axis=0)))
    # print("Std of the test set: {}".format(X_test.std(axis=0)))

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

    # Test of the probabilities
    # y_test_assig_proba = y_test_hat[:, 1] >= 0.5
    # print((y_test_assig == y_test_assig_proba).all())

    # Display confusion matrix when the decision threshold is 0.5
    confm = confusion_matrix(y_true=y_test, y_pred=y_test_assig)
    # plt.figure(1)
    disp = ConfusionMatrixDisplay(confusion_matrix=confm)
    disp.plot()
    plt.title("Confusion Matrix for the logistic regression classifier",
              fontsize=14)
    plt.show()

# %%
# -------------
# PART 2: COMPUTATION AND PLOT OF THE ROC CURVE USING SCIKIT-LEARN
# -------------

    # Calling the functions to build the ROC curve and calculate the AUC
    # Don't use trapz to calculate the ROC curve: There is other option in scikit-learn
    # ====================== YOUR CODE HERE ======================
    fpr, tpr, thresholds = roc_curve(y_test, y_test_hat)
    auc = roc_auc_score(y_test, y_test_assig)
    # ============================================================

    # Plot of the curve
    plt.figure(3)
    plt.plot(fpr, tpr, 'b-', label="ROC of classifier")
    plt.plot([0, 1], [0, 1], 'r--', label="Random classification")
    plt.legend(loc='lower right', shadow=True)
    plt.xlabel("FPR (1-specificity)")
    plt.ylabel("TPR (sensitivity)")
    plt.title("ROC curve calculated with scikit-learn (AUC={:.3f})".
              format(auc))
    plt.xlim([-0.001, 1.001])
    plt.ylim([-0.001, 1.001])
    plt.show()
    