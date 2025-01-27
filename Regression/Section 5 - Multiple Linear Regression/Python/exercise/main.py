# Multiple linear regression
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Dependent Vector

# Taking care of categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# Split dataset for ML model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Compare results y_test with prediction

np.printoptions(precision=2)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))

# Prediction 0,0,1, 112349.2,1123897.8,4124.1
print(regressor.predict([[0, 1, 0, 202349.2, 2023897.8, 20224.1]]))



