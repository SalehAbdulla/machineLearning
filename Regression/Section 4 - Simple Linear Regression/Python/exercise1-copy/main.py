# Simple linear regression
# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Dependent Vector

# Splitting data for ML model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply simple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

#print(y_test)
#print(regressor.predict(X_test))

# Visualise in whole training set

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title("Training set, Income & Experience")
plt.ylabel("Income")
plt.xlabel("Experience")
plt.show()

# Visualise in whole testing set

plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("Training set, Income & Experience")
plt.ylabel("Income")
plt.xlabel("Experience")
plt.show()


# Predict salary of 14 years experience

print("Predict salary of 14 years experience :")
print(regressor.predict([[14]]))

print("Coefficient = ")
print(regressor.coef_)

print("Intercept = ")
print(regressor.intercept_)
