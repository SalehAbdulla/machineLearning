# Importing libraries
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.impute import SimpleImputer

# importing dataset, separating Dependent Vector and Futures
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values  # Future
y = dataset.iloc[:, -1].values  # Dependent Vector

# Taking care of missing data
missingData = dataset.isnull()
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(dataset)
dataset = imputer.transform(dataset)

print(dataset)
