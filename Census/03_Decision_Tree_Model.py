# -*- coding: utf-8 -*-
"""
Project : Census Dataset
Script  : 03_Decision_Tree_Model.py
Purpose : Build a Decision Tree classification model and evaluate
          its predictive performance on the Census dataset.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pickle                                      # Data loading
from sklearn.tree import DecisionTreeClassifier   # Decision Tree classifier
from sklearn.metrics import accuracy_score        # Classification accuracy
from sklearn.metrics import classification_report # Classification performance metrics
from yellowbrick.classifier import ConfusionMatrix # Confusion matrix visualization

# In[ ]: Load the Preprocessed Dataset

# Load the preprocessed Census dataset
with open("census.pkl", mode="rb") as f:

    (
        x_census_treinamento,
        y_census_treinamento,
        x_census_teste,
        y_census_teste
    ) = pickle.load(f)

# Display training and testing set dimensions
x_census_treinamento.shape
y_census_treinamento.shape

x_census_teste.shape
y_census_teste.shape

# In[ ]: Decision Tree

# Create the Decision Tree classifier
tree = DecisionTreeClassifier(
    criterion="entropy",
    random_state=0)

# Train the model
tree.fit(
    x_census_treinamento,
    y_census_treinamento)

# Generate predictions on the test set
prediction = tree.predict(
    x_census_teste)

# Display predictions
prediction

# In[ ]: Model Evaluation

# Calculate test set accuracy
accuracy = accuracy_score(
    y_census_teste,
    prediction)

print("Accuracy:", accuracy)

# Display the confusion matrix
cm = ConfusionMatrix(
    tree)

cm.fit(
    x_census_treinamento,
    y_census_treinamento)

cm.score(
    x_census_teste,
    y_census_teste)

# Display the classification report
print(
    classification_report(
        y_census_teste,
        prediction))

# In[ ]: Conclusion

# The Decision Tree model was trained using the preprocessed
# Census dataset and evaluated on the test set.

# The model achieved an accuracy of approximately 81.04% on the
# test dataset.

# The confusion matrix and classification report provide additional
# information about the model's classification performance across
# the target classes.

# Overall, the Decision Tree provides a classification model for the
# Census dataset that can be evaluated alongside other machine
# learning approaches.
