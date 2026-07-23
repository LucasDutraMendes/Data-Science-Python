# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:28:18 2024

@author: lucas.dutra
"""

#%%

#!pip -q install plotly --upgrade 
#!pip -q install yellowbrick --upgrade  #

import pandas as pd 
import numpy as np
import seaborn as sns #graph visualization
import matplotlib.pyplot as plt #graph visualization
import plotly.express as px #dinamic graph

#%%

        #Loading
#%%

base = pd.read_csv("0-credit_data.csv")

#%%

                    # Exploring & Data Wrangling
#%%
base.head(27)
base.tail(6)
base.describe() # negative min age and count is 1997 indicating missing values

base[base['income'] >= 69995.685578]
base[base['loan'] <= 1.377630]

np.unique(base['default'], return_counts=True) #default counting = 0 not paid, 1 paid 

#inconsistent values = loc pandas function used to search for
base.loc[base['age'] < 0]
base[base['age'] <0] # does the same as the above

"""
to remove one attribute we can use the code below. Not applied for this dataset
#base2 = base.drop('age', axis = 1)

delete only values < 0 - not doing it either
#base3 = base.drop(base[base['age'] < 0].index)
"""
#using the age mean to populate the missing age values
base['age'][base['age'] > 0].mean() #mean not counting on the negative values
base.loc[base['age'] < 0, 'age'] = 40.92

#looking for NA values 
base.isnull().sum()
base.loc[pd.isnull(base['age'])]

#fixing NA values
base['age'].fillna(base['age'].mean(), inplace = True)

#checking the changes
base.loc[(base['clientid'] == 29) | (base['clientid'] == 31) | (base['clientid'] == 32)]
base.loc[base['clientid'].isin([29, 31, 32])]
#%%

        # spliting data frame into class and prediction 
#%%
x_credit = base.iloc[:,1:4].values
x_credit
type(x_credit)

y_credit = base.iloc[:, 4].values
#%%

            #Standardizing 
#%%
"""
#there is a huge difference between the attributes scales. 
#with that being said, we must standardise the them
"""
from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()

x_credit = scaler_credit.fit_transform(x_credit) #applying the transformation 
x_credit

#printing the stardardised values
x_credit[:,0].min(), x_credit[:,1].min(), x_credit[:,2].min()
x_credit[:,0].max(), x_credit[:,1].max(), x_credit[:,2].max()
#%%

            #spliting into training and testing
#%%
from sklearn.model_selection import train_test_split
x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_credit, y_credit, test_size = 0.25, random_state = 0)
x_credit_treinamento.shape
y_credit_treinamento.shape
x_credit_teste.shape
y_credit_teste.shape
#%%

# observing some graphs
#%%
sns.countplot(x = base['default']);
plt.hist(x = base['age']);
plt.hist(x = base['income']);
plt.hist(x = base['loan']);

grafico = px.scatter_matrix(base, dimensions=['age', 'income', 'loan'], color = 'default')
grafico.show()
#%%

                    #Saving
#%%
import pickle
with open('credit.pkl', mode = 'wb') as f2:
  pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f2)
#%%
