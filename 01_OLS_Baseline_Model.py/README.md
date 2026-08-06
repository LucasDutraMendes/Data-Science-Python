# Advertising Dataset - Multiple Linear Regression

**Status:** 🚧 In Progress

This project explores the **Advertising** dataset using **Multiple Linear Regression (OLS)** in **Python** to analyze the relationship between advertising expenditures and product sales.

The repository documents the complete development of a baseline regression model, including data exploration, correlation analysis, model fitting, variable selection, and regression diagnostics. Additional statistical modeling techniques will be incorporated as the project evolves.

---

## Dataset

The **Advertising** dataset contains advertising investments across three different media channels:

- TV
- Radio
- Newspaper

**Target Variable**

- Sales

---

## Objectives

- Explore the relationships among the predictor variables.
- Build a baseline Multiple Linear Regression (OLS) model.
- Perform Stepwise variable selection.
- Evaluate the assumptions of the regression model.
- Interpret the statistical results.
- Continuously improve the model using additional statistical techniques.

---

## Project Structure

```text
Advertising
│
├── Advertising.csv
├── README.md
│
└── 01_OLS_Baseline_Model.py
```

---

## Current Progress

- ✅ Data import
- ✅ Data cleaning
- ✅ Exploratory Data Analysis (EDA)
- ✅ Correlation matrix
- ✅ Pairwise correlation analysis
- ✅ 3D scatter visualization
- ✅ Baseline Multiple Linear Regression (OLS)
- ✅ Confidence intervals
- ✅ Stepwise variable selection
- ✅ Shapiro-Wilk Normality Test
- ✅ Shapiro-Francia Normality Test
- ✅ Durbin-Watson Autocorrelation Test
- ✅ Variance Inflation Factor (VIF)
- ✅ Breusch-Pagan Heteroscedasticity Test
- 🚧 Additional regression techniques (Coming Soon)

---

## Current Workflow

### 1. Data Preparation

- Import dataset
- Remove unnecessary columns
- Explore the dataset structure
- Generate descriptive statistics

### 2. Exploratory Data Analysis (EDA)

- 3D scatter plot
- Pearson correlation matrix
- Pairwise variable relationships
- Pearson correlation coefficients
- Statistical significance (p-values)

### 3. Multiple Linear Regression (OLS)

- Fit the baseline regression model
- Evaluate model summary
- Estimate confidence intervals

### 4. Variable Selection

- Stepwise regression

### 5. Regression Diagnostics

- Shapiro-Wilk Normality Test
- Shapiro-Francia Normality Test
- Durbin-Watson Autocorrelation Test
- Variance Inflation Factor (VIF)
- Breusch-Pagan Heteroscedasticity Test

---

## Technologies

### Programming Language

- Python

### Main Libraries

- pandas
- seaborn
- matplotlib
- plotly
- statsmodels
- scipy
- statstests

---

## Repository Purpose

This repository is part of my **Machine Learning Portfolio**.

Its purpose is to demonstrate a structured approach to statistical modeling in **Python**, emphasizing:

- Data exploration
- Exploratory Data Analysis (EDA)
- Regression modeling
- Variable selection
- Statistical interpretation
- Regression diagnostics
- Reproducible analysis

The repository will continue to evolve as additional modeling techniques and regression improvements are implemented.

---

## Future Development

The following analyses are planned for future versions of this project:

- Box-Cox Transformation
- Ridge Regression
- LASSO Regression
- Principal Component Analysis (PCA)
- Principal Component Regression (PCR)
- Partial Least Squares (PLS)
- Additional regression diagnostics
- Model comparison
- Model performance evaluation
