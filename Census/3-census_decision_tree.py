# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 09:28:39 2024

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

            #DecisionTree
#%%

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion='entropy', random_state=0)

tree.fit(x_census_treinamento, y_census_treinamento)

prediction = tree.predict(x_census_teste)
prediction
#%%

              # Scores
#%%

from sklearn.metrics import accuracy_score, classification_report

accuracy_score(y_census_teste, prediction) #0.8104401228249745 

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(tree)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)

print(classification_report(y_census_teste, prediction))

#%%