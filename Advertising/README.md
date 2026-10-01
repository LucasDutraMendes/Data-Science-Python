# Advertising Dataset - Multiple Linear Regression

**Status:** ✅ Completed

This project explores the **Advertising** dataset using **Multiple Linear Regression (OLS)** in **Python** to analyze the relationship between advertising expenditures and product sales.

The project documents the development of regression models from exploratory data analysis and correlation analysis through regression diagnostics, transformation strategies, variable selection, and model evaluation.

---

## Dataset

The **Advertising** dataset contains advertising expenditures across three different media channels:

- TV
- Radio
- Newspaper

**Target Variable**

- Sales

---

## Objectives

- Explore the relationships among the variables.
- Build a baseline Multiple Linear Regression (OLS) model.
- Perform Stepwise variable selection.
- Evaluate the main assumptions of the regression model.
- Apply Box-Cox transformations to the dependent and independent variables.
- Apply Yeo-Johnson transformation when required.
- Compare alternative regression specifications.
- Select a more parsimonious model.
- Evaluate the final model using regression diagnostics.
- Interpret the statistical and analytical results.

---

## Project Structure

```text
Advertising/
│
├── Advertising.csv
├── README.md
│
├── 01_OLS_Baseline_Model.py
│   ├── Exploratory Data Analysis (EDA)
│   ├── Descriptive Statistics
│   ├── 3D Scatter Plot
│   ├── Correlation Matrix
│   ├── Pairplot
│   ├── Pearson Correlation
│   ├── Multiple Linear Regression (OLS)
│   ├── Confidence Intervals
│   ├── Stepwise Variable Selection
│   ├── Shapiro-Wilk
│   ├── Shapiro-Francia
│   ├── Durbin-Watson
│   ├── Variance Inflation Factor (VIF)
│   ├── Breusch-Pagan
│   └── Conclusion
│
└── 02_BoxCox_Model_Comparison.py
    ├── Box-Cox Transformation (Dependent Variable)
    ├── Box-Cox Transformation (Independent Variables)
    ├── Yeo-Johnson Transformation
    ├── Full Transformation
    ├── OLS Model Comparison
    ├── Stepwise Variable Selection
    ├── Shapiro-Wilk
    ├── Shapiro-Francia
    ├── Durbin-Watson
    ├── Variance Inflation Factor (VIF)
    ├── Breusch-Pagan
    └── Conclusion
```

## Conclusion

The analysis shows a strong relationship between advertising investment and
Sales in this dataset. TV and Radio advertising were the main variables
associated with higher Sales, while Newspaper advertising added little
additional explanatory value and was removed from the final model.

The final model explained approximately 91% of the variation in Sales,
showing that TV and Radio captured most of the information available in the
dataset.

Overall, the project demonstrates how regression analysis can be used to
identify the advertising channels most relevant to Sales while also
highlighting the importance of evaluating model assumptions before drawing
conclusions from statistical results.
