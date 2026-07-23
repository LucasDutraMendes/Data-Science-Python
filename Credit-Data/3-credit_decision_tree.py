# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:55:45 2024

@author: lucas.dutra
"""
#%%

import pickle
with open('C:\machine learning\Python\credit.pkl', 'rb') as f2:
  x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste = pickle.load(f2)
  
  x_credit_treinamento.shape, y_credit_treinamento.shape
  x_credit_teste.shape, y_credit_teste.shape
  
#%%

            #Decision Tree
#%%

from sklearn.tree import DecisionTreeClassifier
tree_1 = DecisionTreeClassifier(criterion= 'entropy', random_state=0)

tree_1.fit(x_credit_treinamento, y_credit_treinamento)
prediction = tree_1.predict(x_credit_teste) #98,2%
prediction

#%%

         # Scores
#%%
from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_credit_teste, prediction) # Score 0.982

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(tree_1)
cm.fit(x_credit_treinamento, y_credit_treinamento)
cm.score(x_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, prediction))

#%%

         # Printing Decision Tree
#%%
tree_1.classes_
from sklearn import tree
import matplotlib.pyplot as plt 
previsores = ['income', 'age', 'loan']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (20,20))
tree.plot_tree(tree_1, feature_names=prediction, class_names=['0','1'], filled=True);
fig.savefig('DecisionTree.png')
#%%