# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:50:04 2024

@author: lucas.dutra
"""

          #Loading
#%%

import pickle

with open('C:\machine learning\Python/census.pkl', mode = 'rb') as f4:
    x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste = pickle.load(f4)
    
    x_census_treinamento.shape, y_census_treinamento.shape
    x_census_teste.shape, y_census_teste.shape
    
#%%

          # Naive Bayes
#%%

from sklearn.naive_bayes import GaussianNB
naive_census = GaussianNB()
naive_census.fit(x_census_treinamento, y_census_treinamento)
prediction = naive_census.predict(x_census_teste)
prediction

#%%

          # Scores
#%%

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
accuracy_score(y_census_teste, prediction)  # Score 0.4767656090071648

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(naive_census)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)

print(classification_report(y_census_teste, prediction))

#%%