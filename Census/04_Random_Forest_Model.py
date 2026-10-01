# -*- coding: utf-8 -*-
"""
Project : Census Dataset
Script  : 04_Random_Forest_Model.py
Purpose : Build a Random Forest classification model and evaluate
          its predictive performance on the Census dataset.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pickle                                       # Data loading
from sklearn.ensemble import RandomForestClassifier # Random Forest classifier
from sklearn.metrics import accuracy_score         # Classification accuracy
from sklearn.metrics import classification_report  # Classification performance metrics
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

# In[ ]: Random Forest

# Create the Random Forest classifier
random_forest = RandomForestClassifier(
    n_estimators=100,
    criterion="entropy",
    random_state=0)

# Train the model
random_forest.fit(
    x_census_treinamento,
    y_census_treinamento)

# Generate predictions on the test set
prediction = random_forest.predict(
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
    random_forest)

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

# The Random Forest model was trained using the preprocessed
# Census dataset and evaluated on the test set.

# The model achieved an accuracy of approximately 85.08% on the
# test dataset.

# The confusion matrix and classification report provide additional
# information about the model's classification performance across
# the target classes.

# Overall, the Random Forest provides a strong classification benchmark
# for the Census dataset and can be compared with other machine
# learning approaches.
