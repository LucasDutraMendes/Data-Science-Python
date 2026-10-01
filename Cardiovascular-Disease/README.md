# Cardiovascular Disease Dataset - Multinomial Logistic Regression

**Status:** 🚧 In Progress

This project explores the **Cardiovascular Disease** dataset using **Multinomial Logistic Regression** in **Python** to analyze the relationship between demographic, clinical, lifestyle, and cardiovascular variables and cardiovascular disease risk level.

The project documents the development of a multinomial classification model from data cleaning and categorical variable encoding through likelihood ratio tests, significance analysis, odds ratios, multicollinearity assessment, model evaluation, and individual prediction.

---

## Dataset

The **Cardiovascular Disease** dataset contains demographic, clinical, and lifestyle variables used to classify individuals into different cardiovascular disease risk levels.

**Predictor Variables**

- Age
- BMI
- HDL
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Estimated LDL
- Smoking Status
- Diabetes Status
- Physical Activity Level
- Family History of CVD

**Target Variable**

- CVD Risk Level

The **HIGH** risk level is used as the reference category in the multinomial logistic regression model.

---

## Objectives

- Clean and prepare the dataset for statistical modeling.
- Handle missing observations.
- Remove redundant, derived, or potentially problematic variables.
- Encode categorical predictors using dummy variables.
- Build a baseline Multinomial Logistic Regression model.
- Evaluate overall model significance using the Likelihood Ratio Test.
- Evaluate the statistical contribution of individual predictors.
- Calculate Wald statistics, p-values, and Odds Ratios.
- Evaluate multicollinearity using Variance Inflation Factor (VIF).
- Evaluate in-sample classification performance.
- Analyze the confusion matrix, sensitivity, and specificity.
- Perform individual risk-level prediction.
- Interpret the statistical and analytical results.

---

## Project Structure

```text
Cardiovascular-Disease/
│
├── 01_GLM_Baseline_Model.py
│   ├── Data Cleaning
│   ├── Missing Value Treatment
│   ├── Categorical Variable Encoding
│   ├── Multinomial Logistic Regression
│   ├── Likelihood Ratio Test
│   ├── Feature Significance Analysis
│   ├── Odds Ratios
│   ├── Variance Inflation Factor (VIF)
│   ├── Model Evaluation
│   ├── Confusion Matrix
│   ├── Sensitivity
│   ├── Specificity
│   ├── Class Distribution
│   ├── Individual Prediction
│   ├── Variable Interpretation
│   └── Conclusion
│
└── README.md
```

## Conclusion

