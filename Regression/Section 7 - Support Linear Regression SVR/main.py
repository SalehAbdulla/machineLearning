# Support Linear Regression
# import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:-1].values  # Features
y = dataset.iloc[:, 2:].values  # Dependent Vector


# Feature Scaling - Reshaping the data to be at the same range
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)


# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')  # Radial Basis function
regressor.fit(X, y)


# Predict a new result
print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)))

