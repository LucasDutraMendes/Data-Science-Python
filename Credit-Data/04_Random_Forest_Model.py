# -*- coding: utf-8 -*-
"""
Project : Credit Data Dataset
Script  : 04_Random_Forest_Model.py
Purpose : Build a Random Forest classification model and evaluate
          its predictive performance on the Credit Data dataset.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pickle                                        # Data loading

from sklearn.ensemble import RandomForestClassifier  # Random Forest classifier
from sklearn.metrics import accuracy_score          # Classification accuracy
from sklearn.metrics import confusion_matrix        # Confusion matrix
from sklearn.metrics import classification_report   # Classification performance metrics

from yellowbrick.classifier import ConfusionMatrix  # Confusion matrix visualization

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

# In[ ]: Random Forest

# Create the Random Forest classifier
random_forest = RandomForestClassifier(
    n_estimators=40,
    criterion="entropy",
    random_state=0)

# Train the model
random_forest.fit(
    x_credit_treinamento,
    y_credit_treinamento)

# Generate predictions on the test set
prediction = random_forest.predict(x_credit_teste)

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
    random_forest)

cm.fit(
    x_credit_treinamento,
    y_credit_treinamento)

cm.score(
    x_credit_teste,
    y_credit_teste)

# In[ ]: Conclusion

# The Random Forest model was trained using the preprocessed
# Credit Data dataset and evaluated on the test set.

# The model achieved an accuracy of approximately 98.4% on the
# test dataset.

# The confusion matrix and classification report provide additional
# information about the model's classification performance across
# the target classes.

# Overall, the Random Forest provides a strong classification model
# for credit default prediction and can be compared with other
# machine learning approaches.
