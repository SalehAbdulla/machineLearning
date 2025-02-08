# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].valus  # Dependent Vector

# Split dataset for ML model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standard deviation rearrangement
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Apply sklean.neighbors
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30,
                                 p=2, metric='minkowski', metric_params=None, n_jobs=None)

classifier.fit(X_train, y_train)
pred_val = classifier.predict(sc.transform([[30, 87000]]))

