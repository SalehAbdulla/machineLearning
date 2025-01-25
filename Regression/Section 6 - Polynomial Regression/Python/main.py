# Polynomial Regression

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values  # Features
y = dataset.iloc[:, -1].values  # Dependent Vector

# Optional for visualisation  linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Creating Polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

# Creating another linear regression for visualisation, feeding poly_red instead of X
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Prediction for position X = 6.5
#new_X_pred = poly_reg.fit_transform([[6.5]])
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))
