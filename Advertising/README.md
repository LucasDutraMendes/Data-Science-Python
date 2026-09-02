# Advertising Dataset - Multiple Linear Regression

**Status:** ✅ Completed

This project explores the **Advertising** dataset using **Multiple Linear Regression (OLS)** in **Python** to analyze the relationship between advertising expenditures and product sales.

The repository documents the development of regression models from data exploration and correlation analysis through regression diagnostics, transformation strategies, variable selection, and final model evaluation.

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
- Apply Box-Cox and Yeo-Johnson transformations.
- Compare alternative regression specifications.
- Select a more parsimonious final model.
- Interpret the statistical and analytical results.

---

## Project Structure

```text
Advertising
│
├── Advertising.csv
├── README.md
│
├── 01_OLS_Baseline_Model.py
└── 02_BoxCox_Model_Comparison.py

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
