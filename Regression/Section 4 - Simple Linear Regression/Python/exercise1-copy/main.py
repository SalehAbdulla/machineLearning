# Simple Linear Regression


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Dependent Vector

# Split the dataset for model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Apply Simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# print actual data and predicted data
print(f"The actual data is {y_test}")
print(f"The predicted data is {y_pred}")

# Visualize the data - Training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.xlabel("Years")
plt.ylabel("Income")
plt.show()

# Visualize the data - test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.xlabel("Years")
plt.ylabel("Income")
plt.show()

# Predict a single value  // in case we have 14 years of experience, lets predict the income
fourteen_prediction = regressor.predict([[14]])
print(f" For 14 years of experience the income would be = {fourteen_prediction}")

# Getting the final linear regression equation  y = coefficient * X + intercept
print(f"Coefficient is = {regressor.coef_}")
print(f"Intercept is = {regressor.intercept_}")
