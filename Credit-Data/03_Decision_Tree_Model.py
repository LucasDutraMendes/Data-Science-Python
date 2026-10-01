# -*- coding: utf-8 -*-
"""
Project : Credit Data Dataset
Script  : 03_Decision_Tree_Model.py
Purpose : Build a Decision Tree classification model, evaluate
          its predictive performance, and visualize the tree.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pickle                                      # Data loading
import matplotlib.pyplot as plt                   # Data visualization

from sklearn.tree import DecisionTreeClassifier  # Decision Tree classifier
from sklearn.tree import plot_tree                # Decision Tree visualization
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

# In[ ]: Decision Tree

# Create the Decision Tree classifier
tree_1 = DecisionTreeClassifier(
    criterion="entropy",
    random_state=0)

# Train the model
tree_1.fit(
    x_credit_treinamento,
    y_credit_treinamento)

# Generate predictions on the test set
prediction = tree_1.predict(x_credit_teste)

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
cm = ConfusionMatrix(tree_1)

cm.fit(
    x_credit_treinamento,
    y_credit_treinamento)

cm.score(x_credit_teste, y_credit_teste)

# In[ ]: Decision Tree Visualization

# Define predictor names
previsores = [
    "income",
    "age",
    "loan"]

# Display the model classes
tree_1.classes_

# Create the Decision Tree visualization
fig, axes = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(20, 20))

plot_tree(
    tree_1,
    feature_names=previsores,
    class_names=["0", "1"],
    filled=True)

# Save the Decision Tree visualization
fig.savefig(
    "DecisionTree.png")

plt.show()

# In[ ]: Conclusion

# The Decision Tree model was trained using the preprocessed
# Credit Data dataset and evaluated on the test set.

# The model achieved an accuracy of approximately 98.2% on the
# test dataset.

# The confusion matrix and classification report provide additional
# information about the model's classification performance across
# the target classes.

# The Decision Tree was also visualized to provide a clearer view
# of the model's decision structure.

# Overall, the Decision Tree provides a strong classification model
# for credit default prediction and can be compared with other
# machine learning approaches.
