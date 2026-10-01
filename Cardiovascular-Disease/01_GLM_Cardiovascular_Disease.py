# Project: Cardiovascular Disease Dataset
# Script : 01_GLM_Cardiovascular_Disease.py
# Purpose: Build and evaluate a multinomial logistic regression model for
#          cardiovascular disease risk level classification.
# Author : Lucas Dutra Mendes

# In[ ]:

import pandas as pd  # Data manipulation and DataFrame management
import statsmodels.api as sm  # Statistical modeling and statistical tests
import numpy as np  # Numerical computing and array operations
from scipy import stats  # Statistical functions and probability distributions
import statsmodels.formula.api as smf  # Statistical models specified using formula syntax
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Calculates VIF for multicollinearity
from sklearn.metrics import accuracy_score  # Calculates classification accuracy
from scipy.stats import chi2  # Chi-square distribution and related statistical calculations

# In[ ]:

df_cardio_disease = pd.read_csv("Cardiovascular_Disease.csv")
df_cardio_disease.describe()
df_cardio_disease.info()

# In[ ]: Data Wrangling / Cleaning

# Blood Pressure - removed because it is already represented
# by Systolic and Diastolic Blood Pressure.
# Height in meters - removed because Height in cm is already present.
# CVD Risk Score - removed because it was calculated using information
# related to the dependent variable, which could cause data leakage.
# Waist-to-Height Ratio - removed because it is derived from
# Abdominal Circumference and Height, avoiding multicollinearity.
# Blood Pressure Category - removed because it is already represented
# by Systolic and Diastolic Blood Pressure.
# Total Cholesterol - removed due to strong linear dependency
# with other lipid variables.
# The following variables were removed during exploratory model refinement
# after evaluating their statistical contribution to the model:
# Weight, Sex, Fasting Blood Sugar, Height and Abdominal Circumference.

df_disease = df_cardio_disease.drop(
    columns=[
        "Blood Pressure (mmHg)",
        "Height (m)",
        "CVD Risk Score",
        "Waist-to-Height Ratio",
        "Blood Pressure Category",
        "Total Cholesterol (mg/dL)",
        "Weight (kg)",
        "Sex",
        "Fasting Blood Sugar (mg/dL)",
        "Height (cm)",
        "Abdominal Circumference (cm)"])

# Looking for NAs    
df_disease.isna().sum()

# Removing NAs
df_disease = df_disease.dropna()

df_disease.describe() # before removing Nas 1529 entries - after 1131

# Printing categories
df_disease["Smoking Status"].value_counts()
df_disease["Diabetes Status"].value_counts()
df_disease["Physical Activity Level"].value_counts()
df_disease["Family History of CVD"].value_counts()
df_disease["CVD Risk Level"].value_counts()

# In[ ]: Changing Var Categories to Numeric before Dummy - 1

df_disease["Smoking Status"] = np.where(
    df_disease["Smoking Status"] == "N", 0, 1)

df_disease["Diabetes Status"] = np.where(
    df_disease["Diabetes Status"] == "N", 0, 1)

df_disease["Family History of CVD"] = np.where(
    df_disease["Family History of CVD"] == "N", 0, 1)
    
df_disease["Physical Activity Level"] = np.where(
    df_disease["Physical Activity Level"] == "Low",0,
    np.where(df_disease["Physical Activity Level"] == "Moderate",1,2))   

# CVD.Risk.Level must be Factor
df_disease["CVD Risk Level"] = df_disease["CVD Risk Level"].astype("category")

# In[ ]: DUMMIES

# Creating Dummy 
disease_dummies = pd.get_dummies(df_disease,
    columns=["Smoking Status", "Diabetes Status", "Physical Activity Level",
             "Family History of CVD"],
    drop_first=False,
    dtype=int)    

# Selecting reference categories for the dummy variables.
disease_dummies = disease_dummies.drop(columns=["Smoking Status_0"])  # 0 = not smoker 
disease_dummies = disease_dummies.drop(columns=["Diabetes Status_0"]) # 0 = not diabetic 
disease_dummies = disease_dummies.drop(columns=["Physical Activity Level_0"])  # 0 = Low physical activity
disease_dummies = disease_dummies.drop(columns=["Family History of CVD_0"])   # 0 = no family history

# In[ ]: Generalized Linear Model - GLM

# CVD.Risk.Level = HIGH = Reference Y variable
df_disease["CVD Risk Level"] = pd.Categorical(
    df_disease["CVD Risk Level"],
    categories=["HIGH", "INTERMEDIARY", "LOW"])

# converts categorical levels into numerical codes, assigning an integer to 
# each category according to the predefined category order. otherwise it would remain False, True
disease_dummies["CVD_Risk_Level"] = (
    disease_dummies["CVD Risk Level"].cat.codes)

# Multinomial Logistic Regression model using statsmodels
glm_disease = smf.mnlogit(
    """CVD_Risk_Level ~
    Age +
    BMI +
    Q('HDL (mg/dL)') +
    Q('Systolic BP') +
    Q('Diastolic BP') +
    Q('Estimated LDL (mg/dL)') +
    Q('Smoking Status_1') +
    Q('Diabetes Status_1') +
    Q('Physical Activity Level_1') +
    Q('Physical Activity Level_2') +
    Q('Family History of CVD_1')""",
    data=disease_dummies).fit()

glm_disease.summary() # Log-Likelihood: -960.37
glm_disease.aic       # AIC:     1968.7359068274855

# In[ ]: chi2
    
null_model = smf.mnlogit(
    "CVD_Risk_Level ~ 1",
    data=disease_dummies
).fit(disp=False)

lr = 2 * (glm_disease.llf - null_model.llf)
df = glm_disease.params.size - null_model.params.size
pvalue = chi2.sf(lr, df)

print("Chi-square:", lr)  # Likelihood Ratio Chi-Square test for the overall model 350.40821937308283
print("df:", df)          # df: 22
print("p-value:", pvalue) # GLOBAL MODEL SIGNIFICANCE: 6.469461920086911e-61

# In[ ]: LRT - For each variable - Similar as a "Manual" Step-Wise

full_model = glm_disease

variables = {
    "Age": ["Age"],
    "BMI": ["BMI"],
    "HDL": ["Q('HDL (mg/dL)')"],
    "Systolic BP": ["Q('Systolic BP')"],
    "Diastolic BP": ["Q('Diastolic BP')"],
    "Estimated LDL": ["Q('Estimated LDL (mg/dL)')"],
    "Smoking Status": ["Q('Smoking Status_1')"],
    "Diabetes Status": ["Q('Diabetes Status_1')"],
    "Physical Activity Level": [
        "Q('Physical Activity Level_1')",
        "Q('Physical Activity Level_2')"
    ],
    "Family History of CVD": ["Q('Family History of CVD_1')"]}

all_variables = [v for group in variables.values() for v in group]

results = []

for name, remove in variables.items():

    remaining = [v for v in all_variables if v not in remove]

    reduced_model = smf.mnlogit(
        "CVD_Risk_Level ~ " + " + ".join(remaining),
        data=disease_dummies
    ).fit(disp=False)

    chi_square = 2 * (full_model.llf - reduced_model.llf)

    df = full_model.params.size - reduced_model.params.size

    p_value = chi2.sf(chi_square, df)

    results.append([
        name,
        chi_square,
        df,
        p_value])

LRT_results = pd.DataFrame(
    results,
    columns=["Variable", "Chi_Square", "df", "P_value"]).sort_values("Chi_Square",
    ascending=False).reset_index(drop=True)

LRT_results

# All predictors were statistically significant according to
# the Likelihood Ratio Test (p < 0.05).

# In[ ]: Wald Test, P-values, and Odds Ratios
    
wald_z = glm_disease.params / glm_disease.bse
p_values = 2 * stats.norm.sf(np.abs(wald_z))

results = pd.DataFrame({
    "Comparison": np.repeat(
        glm_disease.params.columns,
        glm_disease.params.shape[0]),
    "Variable": np.tile(
        glm_disease.params.index,
        glm_disease.params.shape[1]),
    "Coefficient": glm_disease.params.to_numpy().ravel(order="F"),
    "Std_Error": glm_disease.bse.to_numpy().ravel(order="F"),
    "Wald_z": wald_z.to_numpy().ravel(order="F"),
    "P_value": p_values.ravel(order="F"),
    "Odds_Ratio": np.exp(glm_disease.params.to_numpy().ravel(order="F"))})

results = results.sort_values(["Comparison", "P_value"]).reset_index(drop=True)

print(results.to_string())

# In[ ]: VIF Multicollinearity Test
    
X = disease_dummies.drop(
    columns=["CVD Risk Level", "CVD_Risk_Level"])

X = sm.add_constant(X)

vif_results = pd.DataFrame({
    "Variable": X.columns,
    "VIF": [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])]})

vif_results = vif_results[
    vif_results["Variable"] != "const"]

vif_results    # No evidence of multicollinearity
tolerance = 1 / vif_results["VIF"]
tolerance

# In[ ]: # Accuracy - In-Sample Evaluation

# Predictions are made on the same dataset used to fit the model.
# Therefore, these metrics represent in-sample performance and should not
# be interpreted as out-of-sample predictive performance.

predicted_prob = glm_disease.predict(disease_dummies)
predicted_class = predicted_prob.idxmax(axis=1)

accuracy = accuracy_score(
    disease_dummies["CVD_Risk_Level"],
    predicted_class)

accuracy # 0.6693191865605659

# In[ ]: Confusion Matrix

pd.crosstab(
    predicted_class,
    disease_dummies["CVD_Risk_Level"],
    rownames=["Predicted"],
    colnames=["Actual"])

# Accuracy is 66.93%, but the confusion matrix shows substantial differences
# in Recall/Sensitivity across the three classes.

# Recall = TP / (TP + FN) 
# High             - 451 / (451+91+5) = 0.8245
# Intermediary     - 298 / (113+298+3)= 0.7198
# Low              - 8  /  (78+84+8)  = 0.0471

# High Specificity =   TN = 298+84+8+3=393 (True Negative)
#                      FP = 113+78=191 (False Positive)
#            Specificity  = TN/TN+FP = 393/393+191 = 0.6729

# Inte Specificity =   TN = 451+78+5+8=542 (True Negative)
#                      FP = 91+84=175 (False Positive)
#            Specificity  = TN/TN+FP = 542/542+175 = 0.7559

# Low  Specificity =   TN = 451+113+91+298=953 (True Negative)
#                      FP = 5+3=8 (False Positive)
#            Specificity  = TN/TN+FP = 953/953+8 = 0.9917   

# In[ ]: Class Distribution
    
class_distribution = pd.DataFrame({
    "Count": disease_dummies["CVD Risk Level"].value_counts(),
    "Proportion": disease_dummies["CVD Risk Level"].value_counts(normalize=True)
})

class_distribution

# In[ ]: Predict

new_patient = pd.DataFrame({
    "Age": [30],
    "BMI": [24],
    "HDL (mg/dL)": [50],
    "Systolic BP": [130],
    "Diastolic BP": [80],
    "Estimated LDL (mg/dL)": [110],
    "Smoking Status_1": [0],
    "Diabetes Status_1": [0],
    "Physical Activity Level_1": [0],
    "Physical Activity Level_2": [1],
    "Family History of CVD_1": [0]})

# Predicted probabilities
predicted_prob = glm_disease.predict(new_patient)
predicted_prob

# Predicted probabilities

predicted_class = np.argmax(predicted_prob.values, axis=1)

if predicted_class[0] == 1:
    print("Intermediary")
elif predicted_class[0] == 2:
    print("Low")
else:
    print("High")
    
# In[ ]: Variable Interpretation

# Smoking Status:
# Smokers showed lower odds of being classified as INTERMEDIARY rather than
# HIGH (OR = 0.31), corresponding to approximately 69% lower odds.
# Smokers also showed lower odds of being classified as LOW rather than HIGH
# (OR = 0.43), corresponding to approximately 57% lower odds.
#
# Diabetes Status:
# Individuals with diabetes showed lower odds of being classified as
# INTERMEDIARY rather than HIGH (OR = 0.41), corresponding to approximately
# 59% lower odds. For LOW rather than HIGH, the odds were approximately 49%
# lower (OR = 0.51).
#
# Physical Activity:
# Compared with LOW physical activity, MODERATE activity was associated with
# higher odds of INTERMEDIARY rather than HIGH (OR = 2.92), while HIGH activity
# was associated with approximately 3.22 times the odds.
# For LOW rather than HIGH, MODERATE activity had OR = 2.10 and HIGH activity
# had OR = 2.44.
#
# Estimated LDL:
# Each 1 mg/dL increase in estimated LDL was associated with approximately
# 0.91% lower odds of INTERMEDIARY rather than HIGH (OR = 0.99).
# For LOW rather than HIGH, each 1 mg/dL increase was associated with
# approximately 0.61% lower odds (OR = 0.994).
#
# BMI:
# Each 1-unit increase in BMI was associated with approximately 6.7% lower
# odds of INTERMEDIARY rather than HIGH (OR = 0.93), and approximately 4.2%
# lower odds of LOW rather than HIGH (OR = 0.96).
#
# HDL:
# Each 1 mg/dL increase in HDL was associated with approximately 1.7% higher
# odds of INTERMEDIARY rather than HIGH (OR = 1.017), and approximately 2.5%
# higher odds of LOW rather than HIGH (OR = 1.025).

# In[ ]: Conclusion

# The multinomial logistic regression model was statistically significant,
# indicating that the predictors provided relevant information for distinguishing
# between the three cardiovascular risk levels.

# The model achieved an in-sample accuracy of approximately 66.8%. However,
# the confusion matrix revealed substantial differences in classification
# performance across the three classes. Sensitivity was approximately 82.7%
# for HIGH, 71.4% for INTERMEDIARY, and only 4.7% for LOW.

# Although the model showed very high specificity for the LOW class, its
# sensitivity was extremely limited, indicating that the baseline model had
# considerable difficulty correctly identifying LOW-risk observations.

# Therefore, the next stage of the project will focus on class-balancing
# techniques aimed at improving the model's ability to identify LOW-risk
# observations and achieve better predictive performance for this class.
# SMOTE and SMOTENC will be investigated in subsequent analyses, together with
# train/test evaluation to assess out-of-sample predictive performance.