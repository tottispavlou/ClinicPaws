# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 10:10:08 2023

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
# Analyse dataset

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame
sns.pairplot(result)
plt.show()

#%%
# Plots
X = result.drop(columns=['Disease'])
attributeNames = list(X.columns)
X = X.to_numpy()
y = result['Disease'].squeeze()
classNames = list(set(y))

X_c = X.copy()
y_c = y.copy()
attributeNames_c = attributeNames.copy()
i = 0; j = 2;
color = ['r','g', 'b','black']
plt.title('Classification problem')
for c in range(len(classNames)):
    idx = y_c == classNames[c]
    plt.scatter(x=X_c[idx, i],
                y=X_c[idx, j], 
                c=color[c], 
                s=50, 
                alpha=0.8,
                label=classNames[c])
plt.legend()
plt.xlabel(attributeNames_c[i])
plt.ylabel(attributeNames_c[j])
plt.show()

#%%
rr = result.iloc[:, list(range(1,67))]


#%%

from scipy.linalg import svd

# Subtract mean value from data
Y = X - np.ones((60,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#%%
from sklearn.metrics import jaccard_score
import numpy as np

# Assuming 'sparse_matrix' is your sparse binary matrix (a Scipy sparse matrix)
# Convert the sparse matrix to a dense NumPy array to calculate Jaccard similarity
#dense_matrix = sparse_matrix.toarray()
dense_matrix = X[:, 1:66]

# Calculate the Jaccard similarity for each pair of columns (variables)
num_columns = dense_matrix.shape[1]
correlations = np.zeros((num_columns, num_columns))

for i in range(num_columns):
    for j in range(i + 1, num_columns):
        jaccard = jaccard_score(dense_matrix[:, i], dense_matrix[:, j])
        correlations[i, j] = jaccard
        correlations[j, i] = jaccard  # Since Jaccard is symmetric

# 'correlations' is now a matrix where correlations[i, j] represents the Jaccard similarity between columns i and j

import seaborn as sns
import matplotlib.pyplot as plt

# You may want to set up the figure size for better visualization
plt.figure(figsize=(8, 6))

# Create a heatmap
sns.heatmap(correlations, annot=False, cmap='coolwarm', linewidths=0.5, fmt=".2f")
# Add labels and title
plt.title('Correlation Heatmap')
plt.xlabel('Attributes')
plt.ylabel('Attributes')
plt.show() # Show the heatmap


