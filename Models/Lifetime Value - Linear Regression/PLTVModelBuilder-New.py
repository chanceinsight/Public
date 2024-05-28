# -*- coding: utf-8 -*-
"""
Created on Tue May 28 09:29:07 2024

@author: Admin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv("customerltvdata.csv")
data = data[~data.CustomerKey.isin({1, 0})]  #use tilde with isin for is not in

# Preprocess the data
data = data.dropna()
data = pd.get_dummies(data, drop_first=True)

# Define features and target variable
X = data[['Age', 'Gender', 'NumberCarsOwned', 'HouseOwnerFlag', 'YearlyIncome', 'MaritalStatus', 'FirstPurchaseAmount']]  # Replace with actual features
y = data['LifetimeSales']  # Replace with the target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_predictions = model.predict(X_train)
y_test_predictions = model.predict(X_test)

# Evaluate the model
mse_train = mean_squared_error(y_train, y_train_predictions)
mse_test = mean_squared_error(y_test, y_test_predictions)

r2 = r2_score(y_test, y_test_predictions)

print("Train MSE:", mse_train)
print("Test MSE:", mse_test)

print(f'R-squared: {r2}')

# Step 5: Interpret the model coefficients
coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print(coefficients)

# Analyze the results
plt.scatter(y_test, y_test_predictions)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()

sns.residplot(x=y_test, y=y_test_predictions, lowess=True)
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Actual Values')
plt.show()
