# Census Dataset - Classification Models

**Status:** ✅ Completed

This project explores the **Census** dataset using different machine learning classification models in **Python** to predict income categories based on demographic and socioeconomic characteristics.

The project covers the complete workflow from data exploration and preprocessing through feature encoding, model training, evaluation, and comparison of multiple classification algorithms.

---

## Dataset

The **Census** dataset contains demographic and socioeconomic information used to classify individuals according to their income level.

**Target Variable**

- Income

**Examples of Predictor Variables**

- Age
- Workclass
- Education
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Education Number
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

---

## Objectives

- Explore the Census dataset and its main characteristics.
- Identify and inspect missing values.
- Analyze the distribution of the target variable.
- Encode categorical variables.
- Apply feature standardization.
- Split the dataset into training and testing sets.
- Build and evaluate different classification models.
- Compare model performance using accuracy and classification metrics.
- Analyze confusion matrices and class-level performance.
- Identify suitable machine learning approaches for the dataset.

---

## Project Structure

```text
Census/
│
├── 01_Data_Preprocessing.py
│   ├── Dataset Loading
│   ├── Exploratory Data Analysis (EDA)
│   ├── Missing Value Check
│   ├── Class Distribution Analysis
│   ├── Data Visualization
│   ├── Label Encoding
│   ├── One-Hot Encoding
│   ├── Feature Standardization
│   ├── Train/Test Split
│   └── Data Serialization
│
├── 02_Naive_Bayes_Model.py
│   ├── Data Loading
│   ├── Gaussian Naive Bayes
│   ├── Model Evaluation
│   ├── Confusion Matrix
│   ├── Classification Report
│   └── Conclusion
│
├── 03_Decision_Tree_Model.py
│   ├── Data Loading
│   ├── Decision Tree
│   ├── Model Evaluation
│   ├── Confusion Matrix
│   ├── Classification Report
│   └── Conclusion
│
├── 04_Random_Forest_Model.py
│   ├── Data Loading
│   ├── Random Forest
│   ├── Model Evaluation
│   ├── Confusion Matrix
│   ├── Classification Report
│   └── Conclusion
│
└── README.md
