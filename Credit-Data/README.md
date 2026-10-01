# Credit Data Dataset - Classification Models

**Status:** ✅ Completed

This project explores the **Credit Data** dataset using different machine learning classification models in **Python** to predict credit default based on customer characteristics and loan information.

The project covers the complete workflow from exploratory data analysis and data preprocessing through feature standardization, model training, evaluation, and comparison of multiple classification approaches.

---

## Dataset

The **Credit Data** dataset contains information about customers and their credit characteristics.

**Predictor Variables**

- Age
- Income
- Loan

**Target Variable**

- Default

---

## Objectives

- Explore the Credit Data dataset and its main characteristics.
- Identify and treat inconsistent and missing age values.
- Analyze the distribution of the target variable.
- Standardize the predictor variables.
- Split the dataset into training and testing sets.
- Build and evaluate different classification models.
- Compare model performance using classification accuracy and evaluation metrics.
- Analyze confusion matrices and classification reports.
- Explore rule-based classification using CN2.
- Compare different machine learning approaches for credit default prediction.

---

## Project Structure

```text
Credit-Data/
│
├── 01_Data_Preprocessing.py
│   ├── Dataset Loading
│   ├── Exploratory Data Analysis (EDA)
│   ├── Data Cleaning
│   ├── Missing Value Treatment
│   ├── Class Distribution Analysis
│   ├── Feature Selection
│   ├── Feature Standardization
│   ├── Train/Test Split
│   ├── Data Visualization
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
│   ├── Decision Tree Visualization
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
├── 05_CN2_Rule_Learning.py
│   ├── Data Loading
│   ├── Train/Test Split
│   ├── CN2 Rule Learning
│   ├── Rule Inspection
│   ├── Model Prediction
│   ├── Model Evaluation
│   └── Conclusion
│
└── README.md
```

## Model Evaluation

The models were evaluated using test data.

| Model | Accuracy |
|---|---:|
| Gaussian Naive Bayes | 93.8% |
| Decision Tree | 98.2% |
| Random Forest | 98.4% |
| CN2 | 97.4% |

Random Forest achieved the highest accuracy among the evaluated models, followed closely by Decision Tree and CN2.

---

## Conclusion

The analysis showed that customer age, income, and loan information can be used to classify credit default outcomes with high accuracy.

The different classification approaches achieved strong results, with **Random Forest reaching 98.4% accuracy**, followed by **Decision Tree at 98.2%**, **CN2 at 97.4%**, and **Gaussian Naive Bayes at 93.8%**.

Overall, the project demonstrates how different machine learning approaches can be applied to credit classification problems and compared to identify effective solutions for predicting default risk.

---

## Author

**Lucas Dutra Mendes**
