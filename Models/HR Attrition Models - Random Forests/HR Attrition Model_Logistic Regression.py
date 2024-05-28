# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:00:43 2024

@author: Admin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

# Initialize the Logistic Regression classifier
log_reg_classifier = LogisticRegression(max_iter=1000, random_state=42)

# Train the classifier on the training data
log_reg_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = log_reg_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
