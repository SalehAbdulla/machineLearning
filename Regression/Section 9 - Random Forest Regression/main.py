# Random forest Regression
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values  # Features
y = dataset.iloc[:, -1:].values  # Dependent Vector

# implementation of Random Forest Regression
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=10, random_state=0)
regressor.fit(X, y)
#print(regressor.predict([[6.5]]))

# visualisation
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')

plt.show()
