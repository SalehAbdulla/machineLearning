# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values  # Dependent Vector
#y = y.reshape(len(y), 1)

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Applying Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Compare prediction result with real result
y_pred = classifier.predict(X_test)
np.printoptions(precision=2)
#print(np.concatenate((y_test.reshape(len(y_test), 1), y_pred.reshape(len(y_pred), 1)), 1))

# For single prediction
pred_single = sc.transform([[30, 87000]])
prediction = classifier.predict(pred_single)
#print(prediction)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score = accuracy_score(y_test, y_pred)
print(cm)
print(accuracy_score)

# Visualisation
