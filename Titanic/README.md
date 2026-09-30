# Titanic Survival Analysis

## Description

This project applies binary logistic regression to the Titanic dataset
to analyze which passenger characteristics were associated with survival
and to evaluate the model's predictive performance.

The analysis focuses on passenger class, sex, age, and number of siblings
or spouses aboard, using these characteristics to identify differences
in survival outcomes across passenger profiles.

## Topics Covered

- Data Preparation
- Missing Data Treatment
- Categorical Variables
- Dummy Variables
- Binary Logistic Regression
- Stepwise Variable Selection
- Model Evaluation
- Accuracy
- Sensitivity and Specificity
- ROC Curve
- AUC and Gini
- Multicollinearity
- VIF and Tolerance
- Cook's Distance
- Model Prediction

## Dataset

The project uses the Titanic dataset, containing information about
891 passengers and 12 variables.

The analysis uses the following variables:

- `Survived` - survival outcome
- `Pclass` - passenger class
- `Sex` - passenger sex
- `Age` - passenger age
- `SibSp` - number of siblings or spouses aboard
- `Parch` - number of parents or children aboard

## Analysis

### Data Preparation

Missing values in `Age` were replaced using the mean age.

Categorical variables such as `Sex` and `Pclass` were converted into
dummy variables for use in the logistic regression model.

Male passengers and third-class passengers were used as the reference
categories.

### Logistic Regression

A binary logistic regression model was developed to estimate the
probability of survival.

Stepwise variable selection was then applied to obtain a more
parsimonious model.

The final model retained:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`

## Model Evaluation

The final model achieved:

- AUC: `0.899`
- Gini: `0.798`
- Maximum training accuracy: `0.8561236623067776`
- Classification cutoff: `0.67`
- Sensitivity: `0.687096`
- Specificity: `0.9548022`

VIF and Tolerance values showed no evidence of problematic
multicollinearity.

## Key Findings

The analysis indicates that survival was not evenly distributed across
passenger profiles.

Female passengers and passengers from higher classes were more likely
to survive, while increasing age and the number of siblings or spouses
aboard were associated with lower survival probability, holding the
other variables constant.

These results show that demographic and socioeconomic characteristics
in the dataset were strongly associated with different survival
outcomes.

## Example Prediction

The final model can also be used to estimate survival probability for
individual passenger profiles.

| Passenger | Predicted Probability | Classification |
|-----------|----------------------:|----------------|
| Jack      | 7.599%                | Non-survivor   |
| Rose      | 97.55%                | Survivor       |

The classification uses a probability cutoff of `0.67`.

## Conclusion

The analysis shows that survival on the Titanic was strongly related to
passenger profile.

Women and passengers traveling in higher classes had higher survival
probabilities, while older passengers and those traveling with more
siblings or spouses had lower predicted survival probabilities.

The model performed well at distinguishing between survivors and
non-survivors, achieving an AUC of 0.899.

We also checked the influence of individual observations. Although some
passengers had a noticeable impact on the model estimates, the main
relationships remained consistent when these observations were excluded.

Overall, the analysis shows that survival was far from evenly distributed
across passenger profiles and demonstrates how statistical modeling can
turn passenger data into meaningful insights.

## Files

- `01_Titanic_GLM.py` - Data preparation, logistic regression,
  model evaluation, diagnostics, and predictions.
- `titanic.csv` - Titanic dataset.

## Author

Lucas Dutra Mendes

