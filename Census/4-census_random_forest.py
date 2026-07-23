# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:44:52 2024

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

                #RadomForest
#%%

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
random_forest.fit(x_census_treinamento, y_census_treinamento)

prediction = random_forest.predict(x_census_teste)
prediction

#%%

                 # Scores
#%%

from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_census_teste, prediction) # 0.8507676560900717

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_forest)
cm.fit(x_census_treinamento, y_census_treinamento)
cm.score(x_census_teste, y_census_teste)

print(classification_report(y_census_teste, prediction))

#%%