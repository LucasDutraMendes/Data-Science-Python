# Auto MPG Dataset - Multiple Linear Regression

**Status:** 🚧 In Progress

This project explores the **Auto MPG** dataset using **Multiple Linear Regression (OLS)** in **Python** to analyze the relationship between vehicle characteristics and fuel efficiency.

The project documents the development of regression models from data cleaning and exploratory analysis through categorical variable encoding, regression diagnostics, variable selection, and transformation strategies.

---

## Dataset

The **Auto MPG** dataset contains technical and historical characteristics of vehicles used to analyze fuel efficiency.

**Predictor Variables**

- Cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model Year
- Origin

**Target Variable**

- MPG

---

## Objectives

- Explore the relationships among vehicle characteristics.
- Clean and prepare the dataset for statistical modeling.
- Handle missing horsepower values.
- Encode the categorical `origin` variable using dummy variables.
- Build a baseline Multiple Linear Regression (OLS) model.
- Perform Stepwise variable selection.
- Evaluate the main assumptions of the regression model.
- Apply Box-Cox transformation to the dependent variable.
- Compare alternative regression specifications.
- Evaluate multicollinearity, heteroscedasticity, normality, and autocorrelation.
- Perform an individual vehicle prediction.
- Interpret the statistical and analytical results.

---

## Project Structure

```text
Auto-MPG/
│
├── auto-mpg.csv
├── README.md
│
├── 01_OLS_Baseline_Model.py
│   ├── Data Cleaning
│   ├── Missing Value Treatment
│   ├── Correlation Matrix
│   ├── Categorical Variable Encoding
│   ├── Multiple Linear Regression (OLS)
│   ├── Confidence Intervals
│   ├── Stepwise Variable Selection
│   ├── Shapiro-Francia
│   ├── Durbin-Watson
│   ├── Variance Inflation Factor (VIF)
│   ├── Breusch-Pagan
│   ├── Individual Prediction
│   └── Conclusion
│
└── 02_BoxCox_Model_Comparison.py
    ├── ######## NOT FINISHED YET ###########
