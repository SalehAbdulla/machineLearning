# Multiple linear regression
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Dependent Vector

# Taking care of categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)

# Splitting data for ML model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=0)

# Applying linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Compare y_pred & y_test
y_pred = regressor.predict(X_test)
np.printoptions(precision=2)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))

# predict in particular features AMAZING
print(f" CALIFORNIA  &  RD=160000  & Administration=130000  & Marketing=300000 \n {regressor.predict([[1, 0, 0, 160000, 130000, 300000]])}")

