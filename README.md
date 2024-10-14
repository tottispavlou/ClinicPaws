# ClinicPaws

### Introduction
In the realm of veterinary medicine, diagnosing diverse diseases with overlapping symptoms poses challenges for practitioners. The absence of robust data management systems further complicates accurate and timely diagnoses, hindering the tracking of patients' histories. To address this, a project aimed to develop a prototype, ClinicPaws, focusing on improving the diagnostic process for veterinarians. 

The prototype seeks to alleviate the complexities by identifying pain points in the diagnostic journey and providing a user-friendly platform for accessing and reflecting on animal data. The project incorporates machine learning models to enhance diagnostic accuracy, particularly focusing on fungal diseases due to their unique challenges. Through an iterative approach based on user feedback, the platform integrates essential functionalities and undergoes multiple iterations to ensure user satisfaction. The primary objectives include exploring diagnostic pain points, developing a user-friendly platform, integrating machine learning models, and aligning development with user feedback. The ultimate goal is to enhance veterinary practice, addressing the challenges of time-consuming, costly, and emotionally draining diagnostic processes. A comprehensive usability assessment of ClinicPaws evaluates its design, focusing on user experience and the practical application of machine learning in a veterinary context.

A video demo is provided in the repository with the name **demo_video.mp4**. 


### How to run this application:

You need the following requirements to run on your machine:  
`npm install -g electron`  
`npm install exceljs`  
`npm install chart.js`  
`npm install --global yarn`  
`yarn add pdf-parse`  
`npm install multer`  

Also, you need the following python packages:  
`pip install numpy`  
`pip install pandas=1.5.3`  
`pip install xlrd`  
`pip install openpyxl`  
`pip install scikit-learn=1.2.1`  

Then, to start the application, run in your terminal:  
`npm start`


### About the Machine learning part:

The machine learning model is saved in ml_model/model.pickle and the new predictions are run in ml_model/predict_model.py

The code for the analysis and preprocessing of the used dataset *Fungus_diseases_dataset.xlsx*, the iterative training and testing on different models and the training of the final selected model scripts are in the folder ml_model/train_model.
