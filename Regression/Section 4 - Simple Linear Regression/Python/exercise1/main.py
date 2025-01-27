# Simple linear regression
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # dependent Vector

# Split dataset for ML model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Predict the y_test by taken X_test
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# Concatenate y_pred & y_test
np.printoptions(precision=2)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))

# Visualisation Train set
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train))
plt.title("Income and experience")
plt.show()

# Visualisation test set
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.title("Income and experience")
plt.show()
