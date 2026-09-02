# -*- coding: utf-8 -*-
"""
Project: Advertising Dataset
Script : 02_BoxCox_Model_Comparison.py
Purpose: Compare different Box-Cox and Yeo-Johnson transformation
         strategies, evaluate regression assumptions, and select
         the final OLS regression model.
Author : Lucas Dutra Mendes
"""
# In[ ]
"""
     All required packages were loaded in Script 01_OLS_Baseline_Model.py
"""
# In[ ]: Box-Cox Transformation - Dependent Variable

from scipy.stats import boxcox

# Estimate the optimal Box-Cox lambda for the dependent variable y.
lambda_bc_y, lmbda = boxcox(df_advertising['sales'])
print("Lambda: ",lmbda)

# # Apply the Box-Cox transformation to the dependent variable.
df_y_bc = df_advertising.copy()
df_y_bc['sales'] = lambda_bc_y

# Fit a new OLS regression model using the Box-Cox transformed variable.
bc_y_model = sm.OLS.from_formula(
    "sales ~ TV + radio + newspaper",
    data=df_y_bc
).fit()

# Model Summary
bc_y_model.summary()

# In[ ]: Box-Cox Transformation - Independent Variables
# Box-Cox requires strictly positive values.
# The predictor 'radio' contains 1 zero value and therefore
# cannot be transformed directly using Box-Cox.
# Yeo-Johnson will be used to transform 'radio'

# TV - Box-Cox
tv_bc, lambda_bc_tv = boxcox(df_advertising['TV'])
print("Lambda TV:", lambda_bc_tv)

# Newspaper - Box-Cox
newspaper_bc, lambda_bc_newspaper = boxcox(df_advertising['newspaper'])
print("Lambda Newspaper:", lambda_bc_newspaper)

from sklearn.preprocessing import PowerTransformer
# Radio - Yeo-Johnson
yj = PowerTransformer(method='yeo-johnson', standardize=False)

radio_yj = yj.fit_transform(
    df_advertising[['radio']]
).flatten()

lambda_yj_radio = yj.lambdas_[0]
print("Lambda Radio:", lambda_yj_radio)

# Creating a new DataFrame
df_adver_x_transform = pd.DataFrame({
    'sales': df_advertising['sales'],
    'TV': tv_bc,
    'radio': radio_yj,
    'newspaper': newspaper_bc
})

# Fit a new OLS regression model using the Box-Cox transformed variables.
bc_yj_x_model = sm.OLS.from_formula(
    "sales ~ TV + radio + newspaper",
    data=df_adver_x_transform
).fit()

# Model Summary
bc_yj_x_model.summary()

# In[ ]: Full Transformation

# Replace the original dependent variable with the Box-Cox transformed version
df_adver_x_transform['sales'] = df_y_bc['sales']

# Fit a new OLS regression model using the transformed variables
full_trans_model = sm.OLS.from_formula(
    "sales ~ TV + radio + newspaper",
    data=df_adver_x_transform
).fit()

# Display the model summary
full_trans_model.summary()

# In[ ]: Model Selection - Stepwise

# Fit the model using the Stepwise variable selection procedure
step_model = stepwise(
    bc_yj_x_model,
    pvalue_limit=0.05
)

step_model.summary()

# The Stepwise procedure removed newspaper.

# The final model uses the original dependent variable
# and the transformed TV and radio predictors.

# In[ ]: Shapiro-Wilk test

shapiro(step_model.resid)

# In[ ]: Shapiro-Francia test

shapiro_francia(step_model.resid)

#In[ ]: Durbin-Watson Test

# The Durbin-Watson test was performed to assess
# the independence of the residuals.
#
# H0: There is no first-order autocorrelation.
# H1: There is first-order autocorrelation.

# In[ ]: Durbin-Watson

durbin_watson(step_model.resid)

# In[ ]: Variance Inflation Factor (VIF)

# The Variance Inflation Factor was calculated
# to assess multicollinearity among the explanatory variables.
#
# VIF close to 1: no multicollinearity.
# VIF > 5: possible multicollinearity.
# VIF > 10: strong multicollinearity.

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

# The Breusch-Pagan test was performed to evaluate
# whether the residuals exhibit constant variance.
#
# H0: The residuals are homoscedastic.
# H1: The residuals are heteroscedastic.

het_breuschpagan(step_model.resid, sm.add_constant(step_model.fittedvalues))

# In[ ]: Conclusion

# The transformation analysis improved the overall explanatory power
# of the regression model compared with the baseline specification.

# The Stepwise procedure removed the predictor 'newspaper',
# resulting in a more parsimonious model with virtually the same
# explanatory power.

# The final model retains TV and radio as the main predictors of Sales,
# while newspaper provided little additional explanatory value
# after accounting for the other advertising channels.

# The residual diagnostics show that some OLS assumptions remain
# imperfect even after the transformations, particularly normality
# and heteroskedasticity.

# Overall, the final model provides a strong explanation of Sales variation
# while offering a simpler specification based on TV and radio advertising.
