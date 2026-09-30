# Project: Titanic Dataset
# Script : 01_Titanic_GLM.py
# Purpose: Build the baseline binary logistic regression model and evaluate
#          its initial assumptions.
# Author : Lucas Dutra Mendes

# In[ ]:

import pandas as pd  # Data manipulation and DataFrame management
import seaborn as sns  # Statistical data visualization
import matplotlib.pyplot as plt  # Data visualization and plot creation
import statsmodels.api as sm  # Statistical modeling and statistical tests
import numpy as np  # Numerical computing and array operations
from scipy import stats  # Statistical functions and probability distributions
from statsmodels.iolib.summary2 import summary_col  # Creates customized summaries and comparisons of statistical models
import plotly.graph_objs as go  # Interactive graphs and 3D data visualization
import statsmodels.formula.api as smf  # Statistical models specified using formula syntax, including multinomial logistic regression
from statstests.process import stepwise  # Stepwise variable selection for supported statistical models
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Calculates the Variance Inflation Factor (VIF) to assess multicollinearity
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score  # Model evaluation metrics, including ROC curves, confusion matrices, and AUC
import warnings  # Controls and manages Python warning messages

# In[ ]: Loading Dataset
    
df_titanic = pd.read_csv("titanic.csv")
df_titanic

df_titanic.describe()
df_titanic.info()

# Female passengers had a higher observed survival rate than male passengers.
pd.crosstab(
    df_titanic["Sex"],
    df_titanic["Survived"],
    normalize="index")

pd.crosstab(
    df_titanic["Pclass"],
    df_titanic["Survived"],
    normalize="index")

# In[ ]: Data Wrangling
    
# These Variables are numeric and we must parse them to Factor
df_titanic["Sex"] = df_titanic["Sex"].astype("category")
df_titanic["Pclass"] = df_titanic["Pclass"].astype("category")  

# Looking for NAs       
df_titanic.isna().sum() # Age = 177 and Cabin = 687 NAs

# We can use one of these two values Mean or Median to replace the NAs 
df_titanic["Age"].mean()    # 29.69911764705882
df_titanic["Age"].median()  # 28.0
df_titanic["Age"].std()     # 14.526497332334044

# For this model I am using Mean
df_titanic["Age"] = df_titanic["Age"].fillna(df_titanic["Age"].mean())
# df_titanic["Age"] = df_titanic["Age"].fillna(29.69911764705882) 

df_titanic["Sex"].value_counts()
df_titanic["Pclass"].value_counts()

# In[ ]: Creating Dummy - we will not include neither Male nor Pclass_3 in the GLM model   
    
titanic_dummies = pd.get_dummies(df_titanic,
    columns=["Sex", "Pclass"],
    drop_first=False,
    dtype=int)    

# Removing these variables from the analysis    
titanic_dummies = titanic_dummies.drop(columns=["Name"])
titanic_dummies = titanic_dummies.drop(columns=["PassengerId"])
titanic_dummies = titanic_dummies.drop(columns=["Ticket"])    
titanic_dummies = titanic_dummies.drop(columns=["Fare"])    
titanic_dummies = titanic_dummies.drop(columns=["Cabin"])    
titanic_dummies = titanic_dummies.drop(columns=["Embarked"])    
titanic_dummies = titanic_dummies.drop(columns=["Sex_male"])# Removing Male 
titanic_dummies = titanic_dummies.drop(columns=["Pclass_3"])# Removing Pclass_3

# Men and 3rd-class passengers had lower survival rates in the Titanic dataset.
# They are also the most frequent groups in their respective categories,
# so they will be used as the reference categories for the dummy variables.
# Therefore, we keep Sex_female, Pclass_1, and Pclass_2 in the model.

# In[ ]: Generalized Linear Model - GLM

glm_model = smf.glm(formula= "Survived ~ Age + SibSp + Parch + Sex_female + Pclass_1 + Pclass_2",
                    data=titanic_dummies,
                    family=sm.families.Binomial()).fit()

glm_model.aic            # AIC            = 804.3257100547917
glm_model.summary()      # Log-Likelihood = -395.16

# In[ ]: Step-Wise

step_titanic = stepwise(glm_model, pvalue_limit=0.05)

step_titanic.aic             # AIC            = 802.8391799630747
step_titanic.summary()       # Log-Likelihood = -395.42

# In[ ]: Cook's Distance    

influence = step_titanic.get_influence()
cook = influence.cooks_distance[0]

# Plot
plt.figure(figsize=(10, 5))
plt.vlines(
    x=np.arange(len(cook)),
    ymin=0,
    ymax=cook)

plt.axhline(y=4 / len(df_titanic),
    linestyle="--")

plt.title("Cook's Distance")
plt.xlabel("Observation")
plt.ylabel("Cook's Distance")
plt.show()

# Identify observations above the 4/n threshold
threshold = 4 / len(df_titanic)

influential = np.where(cook > threshold)[0]

print("Number of influential observations:", len(influential))

# In[ ]:  # Sensitivity Analysis

# Remove influential observations
titanic_no_influential = titanic_dummies.drop(
    index=titanic_dummies.index[influential])

# Re-estimate the model
titanic_influential = smf.glm(
    formula="Survived ~ Age + SibSp + Sex_female + Pclass_1 + Pclass_2",
    data=titanic_no_influential,
    family=sm.families.Binomial()
).fit()

titanic_influential.summary() # Log-Likelihood -306.38
titanic_influential.aic       # AIC            624.7632386362475

# In[ ]: 

# First method - Defining the cutoff maximizing Sensitivity x Specificity

# 1 - Get the predicted probabilities from the model
predict_matrix = titanic_influential.fittedvalues

# 2 - Get the actual values corresponding to the observations
y_actual = titanic_dummies.loc[
    titanic_influential.model.data.row_labels,
    "Survived"]

# 3 - Generate ROC curve values
# fpr = False Positive Rate
# tpr = True Positive Rate (Sensitivity)
# thresholds = cutoff values
fpr, tpr, thresholds = roc_curve(
    y_actual,
    predict_matrix)

# 3 - Extract sensitivity values
sensitivity_roc = tpr

# 4 - Calculate specificity
#     Specificity = 1 - False Positive Rate
specificity_roc = 1 - fpr

# 5 - Create a DataFrame with cutoff, sensitivity and specificity
plt_data = pd.DataFrame({
    "cutoff": thresholds,
    "specificity": specificity_roc,
    "sensitivity": sensitivity_roc})

# Visualizing the table
print(plt_data)

# 6 - Plot Sensitivity x Specificity
plt.figure(figsize=(10, 6))

plt.plot(
    plt_data["cutoff"],
    plt_data["specificity"],
    label="Specificity")

plt.plot(
    plt_data["cutoff"],
    plt_data["sensitivity"],
    label="Sensitivity")

plt.xlabel("Cutoff")
plt.ylabel("Sensitivity / Specificity")
plt.title("Sensitivity x Specificity according to Cutoff")

plt.legend()
plt.grid(True)

plt.show()

# Sensitivity x Specificity
# According to the graph, the best cutoff is approximately 0.35

# In[ ]: 
    
# Second Method - Defining the cutoff maximizing Accuracy
# Evaluate accuracy across the cutoff values generated by the ROC curve
accuracy = []

# Actual values for the observations used in titanic_influential
y_actual = titanic_no_influential["Survived"].values

for cutoff in thresholds:

    # Convert probabilities into predicted classes
    pred_class = np.where(predict_matrix >= cutoff, 1, 0)

    # Calculate accuracy
    acc = np.mean(pred_class == y_actual)

    accuracy.append(acc)

# Convert accuracy list to NumPy array
accuracy = np.array(accuracy)

# Find the cutoff with maximum accuracy
best_cutoff = thresholds[np.argmax(accuracy)]

# Maximum accuracy
max_accuracy = np.max(accuracy)

print("Best cutoff:", best_cutoff)
print("Maximum accuracy:", max_accuracy)

# Expected result:
# Best cutoff ≈ 0.67
# Maximum accuracy ≈ 0.8561236623067776
# Sensitivity: 0.5789473684210527
# Specificity: 0.9489981785063752

# In[ ]: # Confusion Matrix

# Generate predicted probabilities
predict_matrix = titanic_influential.predict()

# Define the cutoff found previously
cutoff = best_cutoff # = 0.67

# Convert probabilities into predicted classes
pred_class = np.where(predict_matrix >= cutoff,1,0)

# Generate confusion matrix
# Generate confusion matrix
cm = confusion_matrix(y_actual, pred_class)
cm

# Extract:
# Sensitivity = TP / (TP + FN)
# Specificity = TN / (TN + FP)

tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# cutoff = 0.67
print("Sensitivity:", sensitivity)  # 0.687096
print("Specificity:", specificity)  # 0.9548022

# Cutoff = 0.35:
# Sensitivity: 0.7953216374269005
# Specificity: 0.7668488160291439

# In[ ]: Plot ROC Curve

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_actual, predict_matrix)

# Calculate AUC
auc = roc_auc_score(y_actual, predict_matrix)

# Calculate Gini coefficient
gini = 2 * auc - 1

# Plot ROC curve
plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {auc:.3f} | Gini = {gini:.3f}")

# Reference line
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Sensitivity)")

plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.show()

# -----------------------------------------------------------------------------
# AUC = Area Under the Curve - Gini = (AUC-0.5)/0.5
# -----------------------------------------------------------------------------
print("AUC:", auc)     # 0.8992497418139845
print("Gini:", gini)   # 0.7984994836279691

# I am adding an additional AUC graph at the end of this script just for
# curiosity. I consider it a better version.

# In[ ]: VIF - Tolerance
    
X = titanic_influential.model.exog

# Calculate VIF
vif_data = pd.DataFrame()

vif_data["Variable"] = titanic_influential.model.exog_names
vif_data["VIF"] = [
    variance_inflation_factor(X, i)
    for i in range(X.shape[1])]

vif_data

tolerance = 1 / vif_data["VIF"]
tolerance

# The VIF values for all explanatory variables were below 2, indicating no 
# evidence of relevant multicollinearity among the predictors.

# In[ ]: Jack and Rose Hypothetical passengers inspired from Titanic - Prediction
    
# Jack
jack = pd.DataFrame({
    "Age": [22],
    "SibSp": [0],
    "Pclass_1": [0],
    "Pclass_2": [0],
    "Sex_female": [0]})

# Rose
rose = pd.DataFrame({
    "Age": [17],
    "SibSp": [1],
    "Pclass_1": [1],
    "Pclass_2": [0],
    "Sex_female": [1]})

# Combine passengers
new_passengers = pd.concat(
    [jack, rose],
    ignore_index=True)

# Predict probabilities
predictions = titanic_influential.predict(new_passengers)
predictions

# 0.075991  < 0.67 - predicted as non-survivor
# 0.975544  > 0.67 - Rose was predicted as survivor

#===============================================================================
# Conclusion
#===============================================================================

# The analysis shows that survival on the Titanic was strongly related to
# passenger profile.

# Women and passengers traveling in higher classes had higher survival
# probabilities, while older passengers and those traveling with more
# siblings or spouses had lower predicted survival probabilities.

# The model achieved an AUC of 0.899 and a Gini coefficient of 0.798,
# indicating a strong ability to discriminate between survivors and
# non-survivors in the analyzed dataset.

# Cook's Distance was also used to assess the influence of individual
# observations. Although some passengers had a noticeable impact on the
# model estimates, the main relationships remained consistent when these
# observations were excluded.

# Overall, the analysis demonstrates how binary logistic regression can
# be used to analyze a real-world classification problem, evaluate
# predictive performance, investigate influential observations, and
# estimate survival probabilities for individual passenger profiles.
