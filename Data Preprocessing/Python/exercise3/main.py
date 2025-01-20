# Import necessary libraries
import pandas as pd
# Load the Iris dataset
dataset = pd.read_csv("iris.csv")

# Separate features and target
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into an 80-20 training-test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply feature scaling on the training and test sets
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
test = scaler.fit(X_test)  # This wil compute the mean and std
test2 = scaler.transform(X_test)
print(test2)