# -*- coding: utf-8 -*-
"""
Project : Auto MPG Dataset
Script  : 01_OLS_Baseline_Model.py
Purpose : Build the baseline multiple linear regression model and evaluate
          the main regression assumptions.
Author  : Lucas Dutra Mendes
"""

# In[ ]: Import Required Packages

import pandas as pd                     # Data manipulation
import seaborn as sns                   # Statistical data visualization
import matplotlib.pyplot as plt         # Data visualization
import statsmodels.api as sm            # Statistical modeling
import plotly.graph_objects as go       # Interactive 3D visualization
import plotly.io as pio                 # Plotly renderer
import numpy as np                      # Numerical computing
from scipy.stats import pearsonr        # Pearson correlation coefficient
from scipy.stats import shapiro         # Shapiro-Wilk normality test

from statstests.process import stepwise # Stepwise variable selection
from statstests.tests import shapiro_francia  # Shapiro-Francia normality test

from statsmodels.stats.stattools import durbin_watson              # Durbin-Watson autocorrelation test
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Variance Inflation Factor (VIF)
from statsmodels.stats.diagnostic import het_breuschpagan          # Breusch-Pagan heteroscedasticity test

# In[ ]: Load the Dataset

df_auto_mpg = pd.read_csv("auto-mpg.csv")
df_auto_mpg

df_auto_mpg.describe()
df_auto_mpg.info()

df_auto_mpg.isna().any().any()

df_auto_mpg["horsepower"].unique()
(df_auto_mpg["horsepower"] == "?").sum()

# Display rows where horsepower is "?"
df_auto_mpg[df_auto_mpg["horsepower"] == "?"]

# Convert "?" to NaN and convert the column to numeric
df_auto_mpg["horsepower"] = pd.to_numeric(
    df_auto_mpg["horsepower"].replace("?", np.nan)
)

# Missing horsepower values were manually imputed using manufacturer
# specifications obtained from publicly available historical vehicle information

df_auto_mpg.loc[df_auto_mpg["car name"] == "ford pinto", "horsepower"] = 100
df_auto_mpg.loc[df_auto_mpg["car name"] == "ford maverick", "horsepower"] = 84
df_auto_mpg.loc[df_auto_mpg["car name"] == "renault lecar deluxe", "horsepower"] = 51
df_auto_mpg.loc[df_auto_mpg["car name"] == "ford mustang cobra", "horsepower"] = 118
df_auto_mpg.loc[df_auto_mpg["car name"] == "renault 18i", "horsepower"] = 81
df_auto_mpg.loc[df_auto_mpg["car name"] == "amc concord dl", "horsepower"] = 125

# Rename column before modeling
df_auto_mpg = df_auto_mpg.rename(columns={"model year": "model_year"})

# In[ ]: Correlation Matrix

# Compute the correlation matrix
corr = df_auto_mpg.drop(columns=["mpg"]).corr(numeric_only=True)
corr

# Plot the correlation matrix
plt.figure(figsize=(15, 10))

sns.heatmap(
    corr,
    annot=True,
    cmap=plt.cm.viridis,
    annot_kws={"size": 22}
)

plt.show()

# In[ ]:  Create dummy variables for the categorical predictor 'origin'.
# The most frequent category is automatically used as the reference level.

# Showing categories frequency
df_auto_mpg["origin"].value_counts()
df_auto_mpg["origin"].value_counts(normalize=True)

# Find the most frequent category
reference_category = df_auto_mpg["origin"].mode()[0]

# Create N-1 dummy variables
df_auto_mpg_dummies = pd.get_dummies(
    df_auto_mpg,
    columns=["origin"],
    drop_first=False,
    dtype=int
)

# Remove the dummy corresponding to the most frequent category
df_auto_mpg_dummies = df_auto_mpg_dummies.drop(
    columns=[f"origin_{reference_category}"]
)

# Visualizing
df_auto_mpg_dummies

# In[ ]: Multiple Linear Regression

linear_model = sm.OLS.from_formula(
    "mpg ~ cylinders + displacement + horsepower + weight + acceleration + model_year + origin_2 + origin_3",
    data=df_auto_mpg_dummies
).fit()

# Display the model summary
linear_model.summary()

# Display the 95% confidence intervals for the model coefficients
linear_model.conf_int(alpha=0.05)

# In[ ]: Stepwise Variable Selection

# Fit the model using the Stepwise variable selection procedure
step_model = stepwise(
    linear_model,
    pvalue_limit=0.05
)

# In[ ]: Shapiro-Francia Normality Test

shapiro_francia(linear_model.resid) # p-value - 0.0002575
shapiro_francia(step_model.resid)   # p-value - 0.0002004

# In[ ]: Durbin-Watson Test

durbin_watson(linear_model.resid)
durbin_watson(step_model.resid)    
    
# In[ ]: Variance Inflation Factor (VIF)

X = step_model.model.exog[:, 1:]

vif = pd.DataFrame({
    "Variable": step_model.model.exog_names[1:],
    "VIF": [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]
})

print(vif.round(2))
    
# In[ ]: Breusch-Pagan Test

het_breuschpagan(
    linear_model.resid,
    linear_model.model.exog
) # 1.1752175297363932e-05

het_breuschpagan(
    step_model.resid,
    step_model.model.exog
) # 1.026535758577076e-05
    
# In[ ]: Out of curiosity, predict using my car's data
    
honda_city = pd.DataFrame({
    "displacement": [91.34],
    "horsepower": [124.3],
    "weight": [2610],
    "model_year": [82],
    "origin_2": [0],
    "origin_3": [1]
})

prediction_mpg = step_model.predict(honda_city)

prediction_km_l = prediction_mpg * 0.425

print(prediction_km_l)   # 12.79 km/l

# In[ ]: Conclusion

# Conclusion:

# The Auto MPG dataset was analyzed using a Multiple Linear Regression
# model, followed by Stepwise variable selection and regression diagnostics.

# The Shapiro-Francia test indicated that the residuals are not normally
# distributed for both the full and Stepwise models (p < 0.05).

# The VIF analysis indicated severe multicollinearity among several
# explanatory variables, particularly displacement, horsepower, weight,
# and model_year.

# The Breusch-Pagan test indicated strong evidence of heteroskedasticity
# (p < 0.001), suggesting that the residual variance is not constant.

# Overall, the baseline model captures important relationships between
# vehicle characteristics and fuel efficiency, but the diagnostic results
# indicate that several classical OLS assumptions are not fully satisfied.

# Therefore, additional model refinement and transformation strategies
# will be investigated in the next step.
