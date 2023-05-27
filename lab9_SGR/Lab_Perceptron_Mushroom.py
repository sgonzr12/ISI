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
# Indicate 100 as maximum number of iteratios, tolerance 0.00001 and learning rate equal to 1
# ====================== YOUR CODE HERE ======================
clf = Perceptron(max_iter=100, eta0= 1, tol=0.00001)
# ============================================================
    

# Train the Perceptron
clf.fit(X_train, y_train)

# Compute the outputs for the test set
y_test_assig=clf.predict(X_test)

# Evaluate the Classifier 
# Compute the confusion matrix

cm = confusion_matrix(y_test_assig, y_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix for Mushroom Dataset', fontsize=14)
plt.show()

print('=======================')
print('Train Accuracy:',clf.score(X_train, y_train))
print('Test Accuracy:',clf.score(X_test, y_test))
print('=======================')

















