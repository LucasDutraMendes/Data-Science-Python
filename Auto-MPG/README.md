# Auto MPG Dataset - Multiple Linear Regression

**Status:** 🚧 In Progress

This project explores the **Auto MPG** dataset using **Multiple Linear Regression (OLS)** in **Python** to analyze the relationship between vehicle characteristics and fuel efficiency.

The repository documents the complete development of a baseline regression model, including data preparation, correlation analysis, categorical variable encoding, model fitting, Stepwise variable selection, regression diagnostics, and an individual prediction example. Additional statistical modeling techniques will be incorporated as the project evolves.

---

## Dataset

The **Auto MPG** dataset contains technical and historical information about different vehicle models.

**Predictor Variables**

- Cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model Year
- Origin

**Target Variable**

- MPG (Miles per Gallon)

---

## Objectives

- Explore the relationships among the vehicle characteristics.
- Handle missing values in the `horsepower` variable.
- Encode the categorical `origin` variable using dummy variables.
- Build a baseline Multiple Linear Regression (OLS) model.
- Perform Stepwise variable selection.
- Evaluate the assumptions of the regression model.
- Identify multicollinearity among the explanatory variables.
- Interpret the statistical results.
- Generate an individual MPG prediction using the selected model.
- Continuously improve the model using additional statistical techniques.

---

## Project Structure

```text
Auto MPG
│
├── auto-mpg.csv
├── README.md
│
└── 01_OLS_Baseline_Model.py
