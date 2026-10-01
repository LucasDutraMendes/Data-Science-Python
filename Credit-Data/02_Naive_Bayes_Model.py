# -*- coding: utf-8 -*-
"""
Project : Credit Data Dataset
Script  : 02_Naive_Bayes_Model.py
Purpose : Build a Gaussian Naive Bayes classification model and evaluate
          its predictive performance on the Credit Data dataset.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pickle                                      # Data loading
from sklearn.naive_bayes import GaussianNB        # Gaussian Naive Bayes classifier
from sklearn.metrics import accuracy_score        # Classification accuracy
from sklearn.metrics import confusion_matrix      # Confusion matrix
from sklearn.metrics import classification_report # Classification performance metrics
from yellowbrick.classifier import ConfusionMatrix # Confusion matrix visualization

# In[ ]: Load the Preprocessed Dataset

# Load the preprocessed Credit Data dataset
with open("credit.pkl", mode="rb") as f:

    (
        x_credit_treinamento,
        y_credit_treinamento,
        x_credit_teste,
        y_credit_teste
    ) = pickle.load(f)

# Display training and testing set dimensions
x_credit_treinamento.shape
y_credit_treinamento.shape

x_credit_teste.shape
y_credit_teste.shape

# In[ ]: Gaussian Naive Bayes

# Create the Gaussian Naive Bayes classifier
naive_credit_data = GaussianNB()

# Train the model
naive_credit_data.fit(
    x_credit_treinamento,
    y_credit_treinamento)

# Generate predictions on the test set
prediction = naive_credit_data.predict(
    x_credit_teste)

# Display predictions
prediction

# In[ ]: Model Evaluation

# Calculate test set accuracy
accuracy = accuracy_score(
    y_credit_teste,
    prediction)

print("Accuracy:", accuracy)

# Generate and display the confusion matrix
cm_matrix = confusion_matrix(
    y_credit_teste,
    prediction)

print("Confusion Matrix:")
print(cm_matrix)

# Display the classification report
print(
    classification_report(
        y_credit_teste,
        prediction))

# Display the Yellowbrick confusion matrix
cm = ConfusionMatrix(
    naive_credit_data)

cm.fit(
    x_credit_treinamento,
    y_credit_treinamento)

cm.score(
    x_credit_teste,
    y_credit_teste)

# In[ ]: Conclusion

# The Gaussian Naive Bayes model was trained using the preprocessed
# Credit Data dataset and evaluated on the test set.

# The model achieved an accuracy of approximately 93.8% on the
# test dataset.

# The confusion matrix and classification report provide additional
# information about the model's classification performance across
# the target classes.

# Overall, the Gaussian Naive Bayes model provides a strong baseline
# for credit default classification and can be compared with other
# machine learning approaches.
