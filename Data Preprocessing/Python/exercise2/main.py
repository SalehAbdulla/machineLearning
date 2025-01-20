# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
dataset = pd.read_csv('titanic.csv')

# Specify categorical columns by names
categorical_columns = ["Name", "Sex", "Ticket", "Cabin", "Embarked"]

# Apply OneHotEncoder to the categorical columns using ColumnTransformer
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), categorical_columns)],
    remainder='passthrough'  # Keep the other columns as they are
)

# Transform the dataset
transformed_data = ct.fit_transform(dataset)

# Optional: Convert to a DataFrame for readability
transformed_df = pd.DataFrame(transformed_data)

# Example: Encoding the binary target column 'Survived'
binary_categorical_data = dataset['Survived']
le = LabelEncoder()
binary_categorical_encoded = le.fit_transform(binary_categorical_data)

# Print the transformed feature matrix and binary target

print(transformed_df)

print(binary_categorical_encoded)
