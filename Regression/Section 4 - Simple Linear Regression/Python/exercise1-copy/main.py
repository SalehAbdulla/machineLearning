# Simple linear regression
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1].values  # Feature
y = dataset.iloc[:, -1].values  # Dependent Vector

# Split data for ML model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply Simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict y_test by using X_test and compare it with y_pred
y_pred = regressor.predict(X_test)

np.printoptions(precision=2)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))

# Visualize by using chart -- prediction  TEST SET
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("Income and experience")
plt.xlabel("Income")
plt.ylabel("Experience")
plt.show()


# Visualize by using chart -- prediction  TRAINING SET
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Income and experience")
plt.xlabel("Income")
plt.ylabel("Experience")
plt.show()


# Prediction for particular experience  -- y = C * X + I

print(f"20 years of experience, the income would be = {regressor.predict([[20]])}")
print(f"Coefficient is equal = {regressor.coef_}")
print(f"Intercept is equal = {regressor.intercept_}")
