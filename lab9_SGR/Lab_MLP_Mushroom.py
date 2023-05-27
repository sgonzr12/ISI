# -*- coding: utf-8 -*-
"""
@author: Lab ULE
"""

# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay



# 
# -------------
# MAIN PROGRAM
# -------------


# Load the dataset
df = pd.read_csv("mushrooms.csv")

# Let's examine the dataset 
df.head()
df.info()

# Shape of the dataset
# ====================== YOUR CODE HERE ======================
print("Dataset shape:", df.shape)
# ============================================================




# Visualizing the count of edible and poisonous mushrooms
print ("Examples of each class:", df['class'].value_counts())




#Encode the dataset
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
  
# Create the label vector (y) and descriptor matrix (X)
X = df.drop('class', axis=1)
y = df['class']

#Split train-test data
# Use 30% of the data for test and make the split stratified according to the class
# ====================== YOUR CODE HERE ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify= y )
# ============================================================



#Create the classification model: Perceptron
# Indicate one hidden layer with 3 nodes in the hidden layer and activation function 'relu'
# Maximum number of iteratios 150, tolerance 1e-5
# Optimization algorithm: adam
# ====================== YOUR CODE HERE ======================
clf = MLPClassifier(hidden_layer_sizes= (5), max_iter= 1500, tol= 1e-5, activation = "relu", solver= 'adam')
# ============================================================
    

# Train the MLP
# ====================== YOUR CODE HERE ======================
clf.fit(X_train, y_train)
# ============================================================


# Compute the outputs for the test set
# ====================== YOUR CODE HERE ======================
y_test_assig=clf.predict(X_test)
# ============================================================


# Evaluate the Classifier 
# Compute the confusion matrix
# ====================== YOUR CODE HERE ======================
cm = confusion_matrix(y_test_assig, y_test)
# ============================================================
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Mushroom Dataset', fontsize=14)
plt.show()

print('=======================')
print('Train Accuracy:',clf.score(X_train, y_train))
print('Test Accuracy:',clf.score(X_test, y_test))
print('=======================')

















