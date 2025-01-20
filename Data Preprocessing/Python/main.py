# Data processing 1
# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Dependent vector

# taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Taking care of dummy variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

# Split data for machine learning model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_text = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply standard deviation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)
print(X_train)



