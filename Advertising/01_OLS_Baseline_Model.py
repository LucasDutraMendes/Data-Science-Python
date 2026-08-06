# -*- coding: utf-8 -*-
"""
Project : Advertising Dataset
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
from scipy.stats import pearsonr        # Pearson correlation coefficient
from scipy.stats import shapiro         # Shapiro-Wilk normality test

from statstests.process import stepwise # Stepwise variable selection
from statstests.tests import shapiro_francia  # Shapiro-Francia normality test

from statsmodels.stats.stattools import durbin_watson              # Durbin-Watson autocorrelation test
from statsmodels.stats.outliers_influence import variance_inflation_factor  # Variance Inflation Factor (VIF)
from statsmodels.stats.diagnostic import het_breuschpagan          # Breusch-Pagan heteroscedasticity test

# In[ ]: Load the Dataset

# Load the Advertising dataset
df_advertising = pd.read_csv("Advertising.csv")

# Display the dataset
df_advertising

# Display descriptive statistics
df_advertising.describe()

# Display dataset information
df_advertising.info()

# Remove the index column
df_advertising = df_advertising.drop(columns=["Unnamed: 0"])

# In[ ]:  3D Scatter Plot

pio.renderers.default = "browser"

trace = go.Scatter3d(
    x=df_advertising["TV"],
    y=df_advertising["sales"],
    z=df_advertising["radio"],
    mode="markers",
    marker=dict(
        size=5,
        opacity=0.8
    ),
)

layout = go.Layout(
    margin=dict(l=0, r=0, b=0, t=0),
    width=800,
    height=800,
)

plot_figure = go.Figure(data=[trace], layout=layout)

plot_figure.update_layout(
    scene=dict(
        xaxis_title="TV",
        yaxis_title="Sales",
        zaxis_title="Radio"
    )
)

plot_figure.show()

# In[ ]: Correlation Matrix

# Compute the correlation matrix
corr = df_advertising.drop(columns=["sales"]).corr()

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

# Available color palettes
# sns.color_palette("viridis", as_cmap=True)
# sns.color_palette("magma", as_cmap=True)
# sns.color_palette("inferno", as_cmap=True)
# sns.color_palette("Blues", as_cmap=True)
# sns.color_palette("Greens", as_cmap=True)
# sns.color_palette("Reds", as_cmap=True)

# In[ ]: Variable Distributions, Scatter Plots, Pearson Correlations,
# and Corresponding p-values

# Function to display Pearson correlation coefficients and p-values
def corrfunc(x, y, **kws):

    r, p = pearsonr(x, y)

    ax = plt.gca()

    ax.annotate(
        f"r = {r:.2f}",
        xy=(0.10, 0.90),
        xycoords=ax.transAxes
    )

    ax.annotate(
        f"p = {p:.3f}",
        xy=(0.40, 0.90),
        xycoords=ax.transAxes
    )


plt.figure(figsize=(15, 10))

graph = sns.pairplot(
    df_advertising,
    diag_kind="kde"
)

graph.map(corrfunc)

plt.show()

# In[ ]: Multiple Linear Regression

# Fit the multiple linear regression model
linear_model = sm.OLS.from_formula(
    "sales ~ TV + radio + newspaper",
    data=df_advertising
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

# In[ ]: Shapiro-Wilk & Shapiro-Francia Normality Test

# The Shapiro-Wilk & Francia test was performed to evaluate
# whether the residuals follow a normal distribution.
# H0: The residuals are normally distributed.
# H1: The residuals are not normally distributed.

stat, p = shapiro(step_model.resid)

print(f"Statistic = {stat:.4f}")
print(f"P-value   = {p:.15f}")

if p > 0.05:
    print("\nFail to reject H0.")
    print("Residuals are normally distributed.")
else:
    print("\nReject H0.")
    print("Residuals are not normally distributed.")

# Shapiro-Francia test
shapiro_francia(step_model.resid)

# In[ ]: Durbin-Watson Test

# Although the Advertising dataset is cross-sectional rather than a
# time series, the Durbin-Watson test was performed to assess the
# independence of the residuals.

dw = durbin_watson(step_model.resid)

print(f"Durbin-Watson Statistic = {dw:.4f}")

if 1.5 <= dw <= 2.5:
    print("\nResiduals are independent.")
elif dw < 1.5:
    print("\nEvidence of positive autocorrelation.")
else:
    print("\nEvidence of negative autocorrelation.")

# In[ ]: Variance Inflation Factor (VIF)

# The Variance Inflation Factor (VIF) was calculated to assess
# multicollinearity among the explanatory variables.

X = step_model.model.exog

vif = pd.DataFrame({
    "Variable": step_model.model.exog_names,
    "VIF": [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])
    ]
})

print(vif)

# In[ ]: Breusch-Pagan Test

# The Breusch-Pagan test was performed to evaluate whether the residuals
# exhibit constant variance.

# H0: The residuals are homoscedastic.
# H1: The residuals are heteroscedastic.

lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(
    step_model.resid,
    step_model.model.exog
)

print("Breusch-Pagan Test")
print("-" * 40)
print(f"LM Statistic : {lm_stat:.4f}")
print(f"LM p-value   : {lm_pvalue:.4f}")
print(f"F Statistic  : {f_stat:.4f}")
print(f"F p-value    : {f_pvalue:.4f}")

if lm_pvalue > 0.05:
    print("\nFail to reject H0.")
    print("There is no evidence of heteroscedasticity.")
    print("The assumption of homoscedasticity is satisfied.")
else:
    print("\nReject H0.")
    print("Heteroscedasticity detected.")
    print("The assumption of homoscedasticity is violated.")
    
# In[ ]: