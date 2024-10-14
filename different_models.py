# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 11:42:42 2023

@author: Alejandra
"""

"""
https://www.kdnuggets.com/2023/04/best-machine-learning-model-sparse-data.html
Logistic regression: https://medium.com/@rithpansanga/logistic-regression-and-regularization-avoiding-overfitting-and-improving-generalization-e9afdcddd09d
"""

# Imports
import numpy as np
import pandas as pd
from preprocessing_data import *

# Upload dataset
data = pd.read_excel('Fungus_diseases_dataset.xlsx', sheet_name='All diseases')
result = preprocess_data(data)

#%%
# Description of dataset
print(" Description dataset ------------------------")

# How many data points
print("Data points:", len(result))


# Number of classes
print('Number of classes', set(result['Disease']))

# Number of individuals per class
count_blasto = (result['Disease'] == 'Blastomycosis').sum()
count_histo = (result['Disease'] == 'Histoplasmosis').sum()
count_crypto = (result['Disease'] == 'Cryptococcosis').sum()
count_healthy = (result['Disease'] ==  'Healthy').sum()

print(count_blasto, count_histo, count_crypto, count_healthy)



#%%
# Baseline model
print("\n---------BASELINE -----------------")

X = result.drop(columns=['Disease'])
X = X.to_numpy()
y = result['Disease'].squeeze()


# Prediction: class with majority of cases
print('The class with most instances is "Healthy"')
misclass_rate = sum('Healthy' != y) / float(len(y))
Error = misclass_rate
print('Error of the baseline classifier:', Error)


#%%
# Use a classification tree
from sklearn import model_selection, tree
print("\n---------TREE -----------------")

# Simple holdout-set crossvalidation
test_proportion = 0.3
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion, random_state=10)


# Fit decision tree classifier, Gini split criterion, different pruning levels
dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=8)
dtc = dtc.fit(X_train,y_train)

# Evaluate classifier's misclassification rate over train/test data
y_est_test = np.asarray(dtc.predict(X_test))
y_est_train = np.asarray(dtc.predict(X_train))
misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
Error_test, Error_train = misclass_rate_test, misclass_rate_train

print('Train error:', Error_train)
print('Test error:', Error_test)


#%%
# Classification tree with 
print("\n---------TREE with complexity parameter -----------------")
# Tree complexity parameter - constraint on maximum depth
tc = np.arange(2, 21, 1)

# Simple holdout-set crossvalidation
test_proportion = 0.3
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion, random_state=10)

# Initialize variables
Error_train = np.empty((len(tc),1))
Error_test = np.empty((len(tc),1))

for i, t in enumerate(tc):
    # Fit decision tree classifier, Gini split criterion, different pruning levels
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t)
    dtc = dtc.fit(X_train,y_train)

    # Evaluate classifier's misclassification rate over train/test data
    y_est_test = np.asarray(dtc.predict(X_test))
    y_est_train = np.asarray(dtc.predict(X_train))
    misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
    misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
    Error_test[i], Error_train[i] = misclass_rate_test, misclass_rate_train

# Show the error for each tc
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show  
f = figure()
plot(tc, Error_train*100)
plot(tc, Error_test*100)
xlabel('Model complexity (max tree depth)')
ylabel('Error (%)')
legend(['Error_train','Error_test'])  
show() 

# Performance
from sklearn.metrics import  accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
# Confusion matrix (of the last tree)
y_pred = y_est_test
labels = ['Histoplasmosis', 'Healthy',  'Blastomycosis', 'Cryptococcosis']
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)  

#%%
# Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
print("\n-------- NAIVE BAYES -----------------")

# Build a Gaussian Classifier
model = GaussianNB()

# Model training
model.fit(X_train, y_train)

# Predict Output
predicted = model.predict([X_test[6]])
print("Actual Value:", y_test[6])
print("Predicted Value:", predicted[0])
print(model.predict_proba([X_test[6]]))


# Performance
from sklearn.metrics import  accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score

y_pred = model.predict(X_test)
accuracy= accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

y_est_test = y_pred
misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
print('Test error:', misclass_rate_test) # Test error = 1 - accuracy

# Confusion matrix
import matplotlib.pyplot as plt
labels = ['Histoplasmosis', 'Healthy',  'Blastomycosis', 'Cryptococcosis']
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)


#%%
print("\n--------- KNN with k parameter -----------------")
from sklearn.neighbors import KNeighborsClassifier

k_neighbours = [2,3,4,5,6,7,8,9,10,11,12]
Test_error = []

for k in k_neighbours:
    print("k=",k)
    # Leave one out cross validation
    
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    
    Error_test = np.empty((len(X), 1))
    for i, (train_index, test_index) in enumerate(loo.split(X)):
        #print(f"Fold {i}:")
        #print(f"  Train: index={train_index}")
        #print(f"  Test:  index={test_index}")
    
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # Build a Gaussian Classifier
        #model = GaussianNB()
        model = KNeighborsClassifier(n_neighbors=k)
    
        # Model training
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X_test) 
        Error_test[i] = y_pred == y_test
        
    
    # Performance
    accuracy = np.count_nonzero(Error_test == 1)/len(X)
    print('Accuracy:', accuracy)
    
    Test_error.append( 1 - accuracy)


# Show the error for each k
from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, show  
f = figure()
plot(k_neighbours, [i * 100 for i in Test_error])
xlabel('Model complexity (k neighbours)')
ylabel('Error (%)')
legend(['Error_test'])  
show() 



#%%

# KNeighbours
print("\n--------- KNN with k=2 -----------------")

# Simple holdout-set crossvalidation
from sklearn import model_selection
test_proportion = 0.3
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion, random_state=10)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(X_train, y_train)

# Performance
from sklearn.metrics import  accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score

y_pred = neigh.predict(X_test)
accuracy= accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

y_est_test = y_pred
misclass_rate_test = sum(y_est_test != y_test) / float(len(y_est_test))
misclass_rate_train = sum(y_est_train != y_train) / float(len(y_est_train))
print('Test error:', misclass_rate_test) # Test error = 1 - accuracy

# Confusion matrix
import matplotlib.pyplot as plt
labels = ['Histoplasmosis', 'Healthy',  'Blastomycosis', 'Cryptococcosis']
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)

# Make a prediction
new_pacient = [2., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1.,
       0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
       0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
new_pacient= np.reshape(new_pacient, (1, -1))
print(neigh.predict(new_pacient))
print(neigh.predict_proba(new_pacient))




   

