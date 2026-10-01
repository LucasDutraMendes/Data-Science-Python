# -*- coding: utf-8 -*-
"""
Project : Census Dataset
Script  : 02_Naive_Bayes_Model.py
Purpose : Build a Naive Bayes classification model and evaluate
          its predictive performance on the Census dataset.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pickle                                      # Data loading
from sklearn.naive_bayes import GaussianNB        # Gaussian Naive Bayes classifier
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

# In[ ]: Naive Bayes

# Create the Gaussian Naive Bayes classifier
naive_census = GaussianNB()

# Train the model
naive_census.fit(
    x_census_treinamento,
    y_census_treinamento)

# Generate predictions on the test set
prediction = naive_census.predict(
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
    naive_census)

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

# The Gaussian Naive Bayes model was trained using the preprocessed
# Census dataset and evaluated on the test set.

# The model achieved an accuracy of approximately 47.68% on the
# test dataset.

# The confusion matrix and classification report provide additional
# information about the model's classification performance across
# the target classes.

# Overall, this model provides an initial classification benchmark
# for the Census dataset and can be compared with other machine
# learning approaches in subsequent analyses.
