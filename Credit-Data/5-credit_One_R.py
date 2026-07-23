# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:26:48 2024

@author: lucas.dutra
"""

        #Loading
#%%

import Orange
base = Orange.data.Table("credit_data_regras.csv")
base.domain

#%%

        #Splitting 
#%%

base_split = Orange.evaluation.testing.sample(base, n = 0.25)
base_split[0]
base_split[1]

base_training = base_split[1]
base_test = base_split[0]
len(base_training),len(base_test)

#%%

        #Training
#%%

cn2 = Orange.classification.rules.CN2Learner()
rules = cn2(base_training)

#%%

        #Rules
#%%
for i in rules.rule_list:
    print(i)
#%%

        #Predict
#%%

prediction = Orange.evaluation.testing.TestOnTestData(base_training, base_test, [lambda testdata: rules])
prediction
#%%

        #Accuracy 
#%%

Orange.evaluation.CA(prediction) #97.4%

#%%