# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 12:51:28 2023

@author: Alejandra
"""

import pandas as pd

def preprocess_data(data):
    # Clean dataset
    data = data.drop(columns=['Sex', 'Dog ID'])
    
    # Split the comma-separated categories and use get_dummies
    split_categories = data['Clinical Signs'].str.split(', ')
    dummies_clinical = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    split_categories = data['Respiratory Signs'].str.split(', ')
    dummies_respiratory = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    split_categories = data['Intestinal Signs'].str.split(', ')
    dummies_intestinal = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    split_categories = data['CBC Characteristics'].str.split(', ')
    dummies_CBC = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    split_categories = data['Serum Biochemical Profile'].str.split(', ')
    dummies_serumbiochemical = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    split_categories = data['Skin Lesion'].str.split(', ')
    dummies_skin = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    split_categories = data['CNS Signs'].str.split(', ')
    dummies_cns = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    split_categories = data['Ocular Lesion'].str.split(', ')
    dummies_ocular = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    # add the new dummy variables and delete the categorical 
    result = pd.concat([data, dummies_clinical, dummies_respiratory, dummies_intestinal, dummies_CBC, dummies_serumbiochemical, dummies_skin, dummies_cns, dummies_ocular], axis=1)
    result = result.drop(columns=['Clinical Signs', 'Respiratory Signs','Intestinal Signs','CBC Characteristics','Serum Biochemical Profile', 'Skin Lesion', 'CNS Signs', 'Ocular Lesion'])
    
    # Do one-hot encoding of breed, Ultrasound and skin lesion
    result = pd.get_dummies(result, columns=['Breed', 'Ultrasound'])
    
    # Replace NaN values with 0
    for col in list(result.columns):
        result[col].fillna(0, inplace=True)
    
    return result