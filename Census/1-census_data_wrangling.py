# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:00:12 2023

@author: lucas.dutra
"""
             # Packages Importation
#%%

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px 

#%%

           #Loading the Dataset 
#%%

census = pd.read_csv("0-census.csv") 

#%%

        # Exploration & Plotting some data
#%%

census.describe()

census.isnull().sum() #checking if there are any missing values 

np.unique(census['income'], return_counts =True)  #counting

sns.countplot(x = census['income']); 
plt.hist(x = census['age']);
plt.hist(x = census['education-num'])

#%%

             #Plotting 
#%%

grafico = px.treemap(census, path=['workclass','age'])
grafico.show()

#%%
    
             #Plotting 
#%%

grafico2 = px.parallel_categories(census, dimensions=(['occupation']))
grafico2.show()

#%%

            #Splitting the Dataset to begin with the analyses
#%%

x_census = census.iloc[:, 0:14].values

y_census = census.iloc[:, 14].values

#%%

        # Converting string values into numbers 
#%%

from sklearn.preprocessing import LabelEncoder

#applying
label_encoder_workclass = LabelEncoder()
x_census[:,1] = label_encoder_workclass.fit_transform(x_census[:,1])

label_encoder_education = LabelEncoder()
x_census[:,3] = label_encoder_workclass.fit_transform(x_census[:,3])

label_encoder_marital = LabelEncoder()
x_census[:,5] = label_encoder_workclass.fit_transform(x_census[:,5])

label_encoder_occupation = LabelEncoder()
x_census[:,6] = label_encoder_workclass.fit_transform(x_census[:,6])

label_encoder_relationship = LabelEncoder()
x_census[:,7] = label_encoder_workclass.fit_transform(x_census[:,7])

label_encoder_race = LabelEncoder()
x_census[:,8] = label_encoder_workclass.fit_transform(x_census[:,8])

label_encoder_sex = LabelEncoder()
x_census[:,9] = label_encoder_workclass.fit_transform(x_census[:,9])

label_encoder_country = LabelEncoder()
x_census[:,13] = label_encoder_workclass.fit_transform(x_census[:,13])

x_census[0] # visualizing 

#%%

              #Fixing multiple categories attributes - Dummy
#%%

from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
x_census = onehotencoder_census.fit_transform(x_census).toarray()
x_census[0]
x_census.shape 

#%%

           #standardise attributes
#%%

from sklearn.preprocessing import StandardScaler
scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)
x_census[0]

#%%

         #Splitting into training & testing
#%%

from sklearn.model_selection import train_test_split

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census, y_census, test_size = 0.15, random_state = 0)
x_census_treinamento.shape, y_census_treinamento.shape
x_census_teste.shape, y_census_teste.shape

#%%

          #Saving
#%%

import pickle

with open('credit.pkl', mode = 'wb') as f2:
  pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f2)
    
#%%