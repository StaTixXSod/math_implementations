import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from machine_learning.regression.simple_linear_regression import simple_ols
from machine_learning.functions import *


X, y, coeff = make_regression(100, n_features=1, noise=0.1, coef=True, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
intercept, slope, t_val, p_val = simple_ols(X_train[:, 0], y_train)

model = LinearRegression()
model.fit(X_train, y_train)

print("\n======================= OLS =======================")
print(pd.DataFrame({"Intercept": [intercept], "Slope": [slope], "T value": [t_val], "P value": [p_val]}))
print("\n============ Sklearn and OLS comparing ============")
print(pd.DataFrame({"Initial coefficient": coeff, "OLS slope": slope, "Sklearn coefficient": model.coef_}))

assert np.round(model.coef_, 5) == round(slope, 5)

sk_preds = model.predict(X_test)
my_preds = make_prediction(X_test, intercept, slope)

print("\n========= Prediction residuals comparing ==========")
print(pd.DataFrame({"My / Sklearn resids": residuals(my_preds, sk_preds), 
                    "My / Y_true resids": residuals(my_preds, y_test), 
                    "Sklearn / Y_true resids": residuals(sk_preds, y_test)}))

my_r2 = r2_coefficient_score(my_preds, y_test)
sk_r2 = r2_score(y_test, sk_preds)

print("\n==================== R2 score =====================")
print(pd.DataFrame({"My R2 score": my_r2, 
                    "Sklearn R2 score": sk_r2}))

"""
======================= OLS =======================
   Intercept      Slope   T value       P value
0   0.010029  41.772242  9.353988  2.223563e-14

============ Sklearn and OLS comparing ============
   Initial coefficient  OLS slope  Sklearn coefficient
0              41.7411  41.772242            41.772242

========= Prediction residuals comparing ==========
   My / Sklearn resids  My / Y_true resids  Sklearn / Y_true resids
0         2.218727e-28            0.011559                 0.011559

==================== R2 score =====================
   My R2 score  Sklearn R2 score
0     0.999986          0.999986
"""