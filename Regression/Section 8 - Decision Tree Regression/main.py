# Decision Tree Regression
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Features
y = dataset.iloc[:, 2:].values  # Dependent Vector


# Training the decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeClassifier

regressor = DecisionTreeClassifier()
regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])
print(y_pred)

# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

