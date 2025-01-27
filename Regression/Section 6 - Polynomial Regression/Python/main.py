# Polynomial Regression
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Features
y = dataset.iloc[:, -1].values  # Dependent vector

# for testing only lin_reg
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#plt.scatter(X, y, color='red')
#plt.plot(X, lin_reg.predict(X), color='blue')
#plt.title("Salary prediction using linear regression")
#plt.show()

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)

# Applying polynomial regression in linear regression
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

