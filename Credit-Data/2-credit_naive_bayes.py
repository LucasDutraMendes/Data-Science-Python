# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:34:05 2024

@author: lucas.dutra
"""

                                #Loading
#%%
import pickle 
with open('C:\machine learning\Python\credit.pkl', 'rb') as f2:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f2)
  
  x_credit_treinamento.shape, y_credit_treinamento.shape
  x_credit_teste.shape, y_credit_teste.shape
#%%

                              #Naive Bayes
#%%
from sklearn.naive_bayes import GaussianNB
naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_treinamento, y_credit_treinamento)

prediction = naive_credit_data.predict(x_credit_teste)
prediction
#%%
        
        #Scores
#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
accuracy_score(y_credit_teste, prediction) #93.8% accuracy


from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(naive_credit_data)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, prediction))
#%%