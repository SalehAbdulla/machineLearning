# Simple Linear Regression
# Import libraries
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset for model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply Simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
#print(f"The actual data {y_test}\n")
#print(f"The predicted data {y_pred}\n")

# Visualize the data - Training set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Income and experience (Training set) ")
plt.xlabel("Years of experience")
plt.ylabel("Income")
plt.show()

# Visualize the data - test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Income and experience (test set) ")
plt.xlabel("Years of experience")
plt.ylabel("Income")
plt.show()

# Predict a single value
print(regressor.predict([[12]]))

# Getting the final linear regression equation  y = coefficient * X + intercept
print(regressor.coef_)
print(regressor.intercept_)
