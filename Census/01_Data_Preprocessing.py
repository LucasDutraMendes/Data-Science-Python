# -*- coding: utf-8 -*-
"""
Project : Census Dataset
Script  : 01_Data_Preprocessing.py
Purpose : Explore and preprocess the Census dataset for machine learning,
          including categorical encoding, feature standardization,
          train/test splitting, and data serialization.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pickle                          # Data serialization
import numpy as np                     # Numerical computing
import pandas as pd                    # Data manipulation
import seaborn as sns                  # Statistical data visualization
import matplotlib.pyplot as plt        # Data visualization
import plotly.express as px            # Interactive data visualization

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler  

# In[ ]: Load the Dataset

# Load the Census dataset
census = pd.read_csv("0-census.csv")

# Display descriptive statistics
census.describe()

# Display dataset information
census.info()

# In[ ]: Exploratory Data Analysis (EDA)

# Check for missing values
census.isnull().sum()

# Analyze the distribution of the target variable
np.unique(
    census["income"],
    return_counts=True)

# Display the target variable distribution
sns.countplot(
    x=census["income"])

plt.show()

# Plot age distribution
plt.hist(
    census["age"])

plt.show()

# Plot education level distribution
plt.hist(
    census["education-num"])

plt.show()

# In[ ]: Data Visualization

# Visualize the relationship between workclass and age
grafico = px.treemap(
    census,
    path=["workclass", "age"])

grafico.show()

# Visualize occupation categories
grafico2 = px.parallel_categories(
    census,
    dimensions=["occupation"])

grafico2.show()

# In[ ]: Split Features and Target

# Select predictor variables
x_census = census.iloc[:, 0:14].values

# Select target variable
y_census = census.iloc[:, 14].values

# In[ ]: Label Encoding

# Convert categorical attributes into numerical labels
label_encoder_workclass = LabelEncoder()
x_census[:, 1] = label_encoder_workclass.fit_transform(
    x_census[:, 1])

label_encoder_education = LabelEncoder()
x_census[:, 3] = label_encoder_workclass.fit_transform(
    x_census[:, 3])

label_encoder_marital = LabelEncoder()
x_census[:, 5] = label_encoder_workclass.fit_transform(
    x_census[:, 5])

label_encoder_occupation = LabelEncoder()
x_census[:, 6] = label_encoder_workclass.fit_transform(
    x_census[:, 6])

label_encoder_relationship = LabelEncoder()
x_census[:, 7] = label_encoder_workclass.fit_transform(
    x_census[:, 7])

label_encoder_race = LabelEncoder()
x_census[:, 8] = label_encoder_workclass.fit_transform(
    x_census[:, 8])

label_encoder_sex = LabelEncoder()
x_census[:, 9] = label_encoder_workclass.fit_transform(
    x_census[:, 9])

label_encoder_country = LabelEncoder()
x_census[:, 13] = label_encoder_workclass.fit_transform(
    x_census[:, 13])

# Display the first encoded observation
x_census[0]

# In[ ]: One-Hot Encoding

# Convert categorical variables into dummy variables
onehotencoder_census = ColumnTransformer(
    transformers=[
        (
            "OneHot",
            OneHotEncoder(),
            [1, 3, 5, 6, 7, 8, 9, 13]
        )
    ],
    remainder="passthrough")

x_census = onehotencoder_census.fit_transform(
    x_census).toarray()

# Display the first transformed observation
x_census[0]

# Display the transformed dataset shape
x_census.shape

# In[ ]: Feature Standardization

# Standardize the predictor variables
scaler_census = StandardScaler()

x_census = scaler_census.fit_transform(
    x_census)

# Display the first standardized observation
x_census[0]

# In[ ]: Train/Test Split

# Split the dataset into training and testing sets
x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(
    x_census,
    y_census,
    test_size=0.15,
    random_state=0)

# Display training set dimensions
x_census_treinamento.shape
y_census_treinamento.shape

# Display test set dimensions
x_census_teste.shape
y_census_teste.shape

# In[ ]: Data Serialization

# Save the processed datasets for later modeling
with open("credit.pkl", mode="wb") as f2:

    pickle.dump(
        [
            x_census_treinamento,
            y_census_treinamento,
            x_census_teste,
            y_census_teste
        ],
        f2
    )
