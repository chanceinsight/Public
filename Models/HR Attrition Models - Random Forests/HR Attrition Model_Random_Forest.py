# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:00:43 2024

@author: Admin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
csv_location = 'HR_capstone_dataset.csv'
df = pd.read_csv(csv_location)

# Convert 'salary' into numerical values
salary_map = {'low': 0, 'medium': 1, 'high': 2}
df['salary'] = df['salary'].map(salary_map)

# One-hot encode 'Department'
df = pd.get_dummies(df, columns=['Department'])

# Split the dataset into features (X) and target variable (y)
X = df.drop('left', axis=1)  # Features
y = df['left']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
#We initialize the Random Forest classifier with 100 trees and fit it to the training data.
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)