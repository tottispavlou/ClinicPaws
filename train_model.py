# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:25:11 2023
Week9

@author: Alejandra
"""

# Imports
import numpy as np
import pandas as pd
from preprocessing_data import *

# Upload dataset
data = pd.read_excel('Fungus_diseases_dataset.xlsx', sheet_name='All diseases')
result = preprocess_data(data)

#%%
# Modify

X = result.drop(columns=['Disease'])
X = X.to_numpy()
y = result['Disease'].squeeze()


#%%
# Use a classification tree
from sklearn import model_selection, tree

# Simple holdout-set crossvalidation
test_proportion = 0.3
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion, random_state=10)

# Fit decision tree classifier, Gini split criterion, different pruning levels
dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=8)
dtc = dtc.fit(X_train,y_train)


#%% 
# Save model
import pickle as pkl

file = 'model.pickle'
# Create or open a file with write-binary mode and save the model to it
pkl.dump(dtc, open(file, 'wb'))





