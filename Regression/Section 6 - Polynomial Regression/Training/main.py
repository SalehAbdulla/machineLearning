# Polynomial Regression
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1].values  # Features
y = dataset.iloc[:, -1:].values  # Dependent Vector
X = X.reshape(len(X), 1)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
