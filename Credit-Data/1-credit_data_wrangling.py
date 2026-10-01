# -*- coding: utf-8 -*-
"""
Project : Credit Data Dataset
Script  : 01_Data_Preprocessing.py
Purpose : Explore and preprocess the Credit Data dataset for machine
          learning classification models.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pickle                          # Data serialization
import numpy as np                     # Numerical computing
import pandas as pd                    # Data manipulation
import seaborn as sns                  # Statistical data visualization
import matplotlib.pyplot as plt        # Data visualization
import plotly.express as px            # Interactive data visualization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# In[ ]: Load the Dataset

# Load the Credit Data dataset
base = pd.read_csv("0-credit_data.csv")

# In[ ]: Exploratory Data Analysis (EDA)

# Display the first observations
base.head(27)

# Display the last observations
base.tail(6)

# Display descriptive statistics
base.describe()

# Analyze extreme income values
base[base["income"] >= 69995.685578]

# Analyze very low loan values
base[base["loan"] <= 1.377630]

# Analyze the target variable distribution
np.unique(
    base["default"],
    return_counts=True
)

# In[ ]: Data Cleaning

# Identify inconsistent age values
base.loc[base["age"] < 0]

# Calculate the mean age using only positive values
mean_age = base.loc[
    base["age"] > 0,
    "age"].mean()

print("Mean age:", mean_age)

# Replace negative age values with the mean age
base.loc[
    base["age"] < 0,
    "age"] = mean_age

# Check for missing values
base.isnull().sum()

# Display rows with missing age values
base.loc[
    base["age"].isnull()]

# Replace missing age values with the mean age
base["age"] = base["age"].fillna(
    base["age"].mean())

# Verify the corrections
base.loc[
    base["clientid"].isin([29, 31, 32])]

# In[ ]: Split Features and Target

# Select predictor variables
x_credit = base.iloc[:, 1:4].values

# Select target variable
y_credit = base.iloc[:, 4].values

# In[ ]: Feature Standardization

# Standardize the predictor variables because they have different scales
scaler_credit = StandardScaler()

x_credit = scaler_credit.fit_transform(x_credit)

# Display standardized values
x_credit

# Check standardized feature ranges
x_credit[:, 0].min(), x_credit[:, 1].min(), x_credit[:, 2].min()
x_credit[:, 0].max(), x_credit[:, 1].max(), x_credit[:, 2].max()

# In[ ]: Train/Test Split

# Split the dataset into training and testing sets
x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(
    x_credit,
    y_credit,
    test_size=0.25,
    random_state=0)

# Display training set dimensions
x_credit_treinamento.shape
y_credit_treinamento.shape

# Display test set dimensions
x_credit_teste.shape
y_credit_teste.shape

# In[ ]: Data Visualization

# Plot target variable distribution
sns.countplot(
    x=base["default"])

plt.show()

# Plot age distribution
plt.hist(
    base["age"])

plt.show()

# Plot income distribution
plt.hist(
    base["income"])

plt.show()

# Plot loan distribution
plt.hist(
    base["loan"])

plt.show()

# Explore relationships among the predictor variables
# colored by the target variable
grafico = px.scatter_matrix(
    base,
    dimensions=["age", "income", "loan"],
    color="default")

grafico.show()

# In[ ]: Data Serialization

# Save the preprocessed datasets for later modeling
with open("credit.pkl", mode="wb") as f:

    pickle.dump(
        [
            x_credit_treinamento,
            y_credit_treinamento,
            x_credit_teste,
            y_credit_teste
        ],
        f
    )
