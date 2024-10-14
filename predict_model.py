# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 12:49:25 2023

@author: Alejandra
"""
# Requirements ###############################################################################################################
#pip install numpy
#pip install pandas=1.5.3
#pip install xlrd (maybe not needed)
#pip install openpyxl
#pip install scikit-learn=1.2.1

##############################################################################################################################

import warnings
# Temporarily suppress all warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle as pkl

#%%
# Get info from run_ml.html ##################################################################################################
import sys

try: 
    # Access the command-line arguments
    patient_dir = sys.argv[1]  # The first argument is the path to the patient directory

    # Now you can use the patient_dir variable in your Python script
    #print("Processing data for patient in directory:" + patient_dir)
except:
    print("Error. Patient data not loaded ")


#%%
# Read info about pacient #####################################################################################################
#name_patient = 'Tom'
#phone_number = '123456'
#pacient_ml_data = pd.read_excel('patient_data/'+name_patient+'_'+phone_number+'/patient_ml_data.xlsx', engine='openpyxl')
#patient_info = pd.read_excel('patient_data/'+name_patient+'_'+phone_number+'/patient_info.xlsx', engine='openpyxl')

pacient_ml_data = pd.read_excel('patient_data/'+ patient_dir +'/patient_ml_data.xlsx', engine='openpyxl')
patient_info = pd.read_excel('patient_data/'+ patient_dir +'/patient_info.xlsx', engine='openpyxl')
breed = patient_info.loc[0,'Breed']
age = 2023 - patient_info.loc[0, 'Year of Birth']


# Create dataset to process ##################################################################################################
data = pd.concat([pacient_ml_data, patient_info['Breed']], axis=1)
data['Age'] = [age]

##############################################################################################################################
# Preprocessing

# Where the nan are
nan_columns = data.columns[data.isna().any()].tolist()
no_nan_columns = list(set(data.columns) - set(nan_columns) - set(['Age']))

for c in no_nan_columns:
    # Split the comma-separated categories and use get_dummies
    split_categories = data[c].str.split(', ')
    dummies = pd.get_dummies(split_categories.apply(pd.Series).stack()).sum(level=0)
    
    # add the new dummy variables and delete the categorical 
    result = pd.concat([data, dummies], axis=1)
    result = result.drop(columns=c)

    
if 'Breed' in nan_columns:
    print('Please, enter the breed of the dog')
else:
    # Do one-hot encoding of breed
    result = pd.get_dummies(result, columns=['Breed'])

if 'Ultrasound' in no_nan_columns:
    # Do one-hot encoding of Ultrasound (((and skin lesion)))
    result = pd.get_dummies(result, columns=['Ultrasound'])
    

# Replace NaN values with 0
for col in list(result.columns):
    result[col].fillna(0, inplace=True)

# Resultant preprocessed dataframe
patient_data_pp = result


#%%
# Process by hand ####################################################################################################

# Define the columns you want in your DataFrame
columns = ['Age (Years)', 'Anorexia', 'Depression', 'Fever', 'Lethargy',
       'Weight Loss', 'Cough', 'Cyanosis', 'Dyspnea', 'Pleural Effusion',
       'Respiratory Distress', 'Tachypnea', 'Diarreah',
       'Intestinal Blood Loss', 'Tenesmus', 'Hypercalcemia',
       'Hyperglobulinemia', 'Hypoalbuminemia', 'Mature Neutrophilia',
       'Mild Nonregenerative Anemia', 'Neutrophilia with Left Shift',
       'Nonregenerative Anemia', 'Thrombocytopenia', 'Calcium UP',
       'Hepatic enzymes UP', 'Within Reference Ranges',
       'Peripheral Lymphadenopathy', 'Yes', 'Ataxia',
       'Central Vestibular Disease', 'Cervical Pain',
       'Multifocal Cranial Nerve Involvement', 'Papilledema', 'Seizure',
       'Tetraparesis', 'Granulomatous Chorioretinitis', 'Optic Neuritis',
       'Retinal Hemorrhage', 'Breed_American Cocker Spaniel', 'Breed_Beagle',
       'Breed_Boston Terrier', 'Breed_Boxer', 'Breed_Brittany',
       'Breed_Bulldog', 'Breed_Chihuahua', 'Breed_Cocker Spaniel',
       'Breed_Dachshund', 'Breed_Dalmatian', 'Breed_Doberman',
       'Breed_Doberman Pinscher', 'Breed_German Sheperd',
       'Breed_Golden Retriever', 'Breed_Great Dane',
       'Breed_Labrador Retriever', 'Breed_Mastiff', 'Breed_Pointer',
       'Breed_Pomeranian', 'Breed_Pug', 'Breed_Rottweiler',
       'Breed_Saint Bernard', 'Breed_Shih Tzu', 'Breed_Siberian Husky',
       'Breed_Weimaraners', 'Breed_Wiemaraners', 'Breed_Yorkshire Terrier',
       'Ultrasound_Organomegaly']

# Create an empty DataFrame with the specified columns
df = pd.DataFrame(columns=columns)

# Create a single row with NaN values
nan_values = [np.nan] * len(columns)
df = df.append(pd.Series(nan_values, index=columns), ignore_index=True)


for c in columns:
    try:
        df[c] = int(patient_data_pp[c])
        #print(c, 'is a feature in the dog.')
    except:
        df[c] = 0
        #print(c, 'is not there.')


#%%
# Process the blood work ######################################################################################################

try:
    # Upload dataset
    blood_draft = pd.read_excel('patient_data/'+ patient_dir +'/blood_draft.xlsx', names=['Result', 'Reference values'], engine='openpyxl')
    
    # Create a new column with the test names
    blood_draft['Test'] = ['HCT', 'HGB', 'MVC', 'MCH', 'MCHC', 'RDW','%RETIC',
                           'RETIC', 'RETIC-HGB', 'WBC', '%NEU', '%LYM', '%BASO', '%EOS', 'NEU', 'LYM',
                           'MONO', 'EOS', 'BASO', 'PLT', 'MPV', 'PDW', 'PCT',
                           'GLU', 'CREA', 'BUN', 'BUN/CREA', 'PHOS', 'CA', 'TP', 'ALB', 'GLOB', 'ALB/GLOB', 'ALT',
                           'ALKP', 'GGT', 'TBIL', 'CHOL', 'AMYL', 'LIPA',
                           'Na+', 'K+', 'Cl-', 'Ca++', 'Glu', 'Lac'] 
    
    # Include info about blood work in final dataframe (df)
    if float(blood_draft[blood_draft['Test'] == 'GGT']['Result']) > 11.0:
        df['Hepatic enzymes UP'][0] = int(1)
    if float(blood_draft[blood_draft['Test'] == 'ALT']['Result']) > 125.0:
        df['Hepatic enzymes UP'][0] = int(1)
    if float(blood_draft[blood_draft['Test'] == 'Ca++']['Result']) > 2.60:
        df['Calcium UP'][0] = int(1)

except:
    # if there is no bloodwork file, do nothing
    pass


#%%
# Make a prediction #####################################################################################################

new_pacient = df.to_numpy()
new_pacient = np.reshape(new_pacient, (1, -1))

# Open the saved file with read-binary mode containing the model
model = pkl.load(open('ml_model/model.pickle', 'rb'))

# Use the loaded model to make predictions 
prediction = np.asarray(model.predict(new_pacient))[0]

# If prediction is "Healthy"
if prediction == "Healthy":
    prediction = "Not a fungal disease"

# Print message
print('Prediction for '+ patient_dir +': ', prediction)
