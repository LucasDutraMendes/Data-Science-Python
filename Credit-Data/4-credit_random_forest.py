# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:04:30 2024

@author: lucas.dutra
"""

#%%

import pickle
with open('C:\machine learning\Python\credit.pkl', 'rb') as f2:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f2)
  
  x_credit_treinamento.shape, y_credit_treinamento.shape
  x_credit_teste.shape, y_credit_teste.shape
  
#%%

#%%

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
random_forest.fit(x_credit_treinamento, y_credit_treinamento)

prediction = random_forest.predict(x_credit_teste)
prediction

#%%

#%%

from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_credit_teste, prediction) # 0.984

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_forest)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, prediction))
#%%
