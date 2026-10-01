# -*- coding: utf-8 -*-
"""
Project : Advertising Dataset
Script  : 02_BoxCox_Model_Comparison.py
Purpose : Compare different Box-Cox and Yeo-Johnson transformation
          strategies, evaluate regression assumptions, and select
          the final OLS regression model.
Author  : Lucas Dutra Mendes
"""

# In[ ]:
"""
The required packages and baseline models were loaded
in Script 01_OLS_Baseline_Model.py.
"""

# In[ ]: Box-Cox Transformation - Dependent Variable

from scipy.stats import boxcox

# Estimate the optimal Box-Cox lambda for the dependent variable
sales_bc, lambda_bc_sales = boxcox(
    df_advertising["sales"])

print("Lambda Sales:", lambda_bc_sales)

# Apply the Box-Cox transformation to the dependent variable
df_y_bc = df_advertising.copy()
df_y_bc["sales"] = sales_bc

# Fit an OLS regression model using the transformed dependent variable
bc_y_model = sm.OLS.from_formula(
    "sales ~ TV + radio + newspaper",
    data=df_y_bc).fit()

# Display the model summary
bc_y_model.summary()

# In[ ]: Box-Cox Transformation - Independent Variables

# Box-Cox requires strictly positive values.
# The predictor 'radio' contains a zero value and therefore
# cannot be transformed directly using Box-Cox.
# Yeo-Johnson will be used to transform 'radio'.

# TV - Box-Cox
tv_bc, lambda_bc_tv = boxcox(
    df_advertising["TV"])

print("Lambda TV:", lambda_bc_tv)

# Newspaper - Box-Cox
newspaper_bc, lambda_bc_newspaper = boxcox(
    df_advertising["newspaper"])

print("Lambda Newspaper:", lambda_bc_newspaper)

# Radio - Yeo-Johnson
from sklearn.preprocessing import PowerTransformer

yj = PowerTransformer(
    method="yeo-johnson",
    standardize=False)

radio_yj = yj.fit_transform(
    df_advertising[["radio"]]).flatten()

lambda_yj_radio = yj.lambdas_[0]

print("Lambda Radio:", lambda_yj_radio)

# Create a new DataFrame with the transformed predictors
df_adver_x_transform = pd.DataFrame({
    "sales": df_advertising["sales"],
    "TV": tv_bc,
    "radio": radio_yj,
    "newspaper": newspaper_bc})

# Fit an OLS regression model using the transformed predictors
bc_yj_x_model = sm.OLS.from_formula(
    "sales ~ TV + radio + newspaper",
    data=df_adver_x_transform).fit()

# Display the model summary
bc_yj_x_model.summary()


# In[ ]: Full Transformation

# Replace the original dependent variable with the Box-Cox transformed version
df_adver_x_transform["sales"] = df_y_bc["sales"]

# Fit an OLS regression model using the transformed dependent variable
# and transformed independent variables
full_trans_model = sm.OLS.from_formula(
    "sales ~ TV + radio + newspaper",
    data=df_adver_x_transform).fit()

# Display the model summary
full_trans_model.summary()


# In[ ]: Model Selection - Stepwise

# Apply Stepwise variable selection to the model with transformed predictors
step_model_bc = stepwise(
    bc_yj_x_model,
    pvalue_limit=0.05)

# Display the selected model
step_model_bc.summary()

# The Stepwise procedure removed 'newspaper'.
#
# The final model retains the original dependent variable
# and the transformed TV and radio predictors.


# In[ ]: Shapiro-Wilk & Shapiro-Francia Normality Tests

# The Shapiro-Wilk and Shapiro-Francia tests were performed
# to evaluate whether the residuals follow a normal distribution.
#
# H0: The residuals are normally distributed.
# H1: The residuals are not normally distributed.

shapiro(step_model_bc.resid)

# Shapiro-Francia test
shapiro_francia(step_model_bc.resid)


# In[ ]: Durbin-Watson Test

# The Durbin-Watson test was performed to assess
# the independence of the residuals.
#
# H0: There is no first-order autocorrelation.
# H1: There is first-order autocorrelation.

durbin_watson(step_model_bc.resid)


# In[ ]: Variance Inflation Factor (VIF)

# The Variance Inflation Factor (VIF) was calculated
# to assess multicollinearity among the explanatory variables.
#
# VIF close to 1: little evidence of multicollinearity.
# VIF > 5: possible multicollinearity.
# VIF > 10: strong multicollinearity.

# Exclude the intercept from the VIF calculation
X = step_model_bc.model.exog[:, 1:]

vif = pd.DataFrame({
    "Variable": step_model_bc.model.exog_names[1:],
    "VIF": [
        variance_inflation_factor(X, i)
        for i in range(X.shape[1])]})

print(vif.round(2))


# In[ ]: Breusch-Pagan Test

# The Breusch-Pagan test was performed to evaluate whether the residuals
# exhibit constant variance.
#
# H0: The residuals are homoscedastic.
# H1: The residuals are heteroscedastic.

bp_test = het_breuschpagan(
    step_model_bc.resid,
    step_model_bc.model.exog)

bp_test

# In[ ]: Conclusion

# The Box-Cox and Yeo-Johnson transformations were evaluated
# to improve the specification and regression assumptions
# of the baseline multiple linear regression model.

# The transformed predictor model was subjected to Stepwise
# variable selection. The procedure removed 'newspaper',
# resulting in a more parsimonious specification that retains
# TV and radio as the main explanatory variables.

# The selected model uses the original dependent variable,
# with TV and radio transformed using Box-Cox and Yeo-Johnson,
# respectively.

# The Shapiro-Wilk and Shapiro-Francia tests were used to assess
# the normality of the residuals after transformation.

# The Durbin-Watson statistic was used to evaluate
# first-order autocorrelation among the residuals.

# The Variance Inflation Factor (VIF) was used to assess
# multicollinearity among the explanatory variables.

# The Breusch-Pagan test was used to evaluate
# the homoscedasticity assumption.

# Stepwise variable selection applied to the transformed predictor model
# resulted in a more parsimonious specification; however, the residual
# diagnostics indicate that some classical OLS assumptions may remain
# unsatisfied.

# Overall, the selected model provides a simplified specification
# based on the original Sales variable and the transformed TV and radio
# predictors, while further model refinement may be required to address
# the remaining diagnostic issues.
