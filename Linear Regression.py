# -*- coding: utf-8 -*-
"""
Created on Sat May 18 20:08:06 2024

@author: Chris Bower
Developed with the help of ChatGPT :)

The programme imports a file 'salary_dataset' originated on kaggle.com

The are 2 columns in the file 'YearsExperience' and Salary

The code imports the file into a Pandas dataframe and 
sklearn is used to build the Linear Regression model
This gives us the intercept and slope of the line

Next r squared and the p-value are determined using stats from scipy

The values are printed out and finally the regression line graph is drawn
with Matplotlib.pyplot


"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt

def load_data(csv_location):
    """Load CSV data into a pandas DataFrame."""
    df = pd.read_csv(csv_location)
    return df

def prepare_data(df):
    """Prepare independent and dependent variables for regression."""
    X = df[['YearsExperience']].values  # Extract as 2D array
    y = df['Salary'].values  # Extract as 1D array
    return X, y

def fit_sklearn_model(X, y):
    """Fit a linear regression model using sklearn."""
    model = LinearRegression()
    model.fit(X, y)
    return model

def fit_statsmodels_ols(X, y):
    """Fit a linear regression model using statsmodels OLS."""
    X_with_const = sm.add_constant(X)  # Add an intercept to the model
    model = sm.OLS(y, X_with_const).fit()
    return model

def plot_regression_line(df, intercept, slope):
    """Plot the regression line along with data points."""
    plt.scatter(df['YearsExperience'], df['Salary'], color='blue', label='Data points')
    plt.plot(df['YearsExperience'], intercept + slope * df['YearsExperience'], color='red', label='Regression line')
    plt.title("Salary by Years of Experience")
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.legend()
    plt.show()

def main():
    csv_location = "salary_dataset.csv"
    df = load_data(csv_location)

    # Display basic statistics
    #print(df.describe())

    X, y = prepare_data(df)

    # Fit the model using sklearn
    sklearn_model = fit_sklearn_model(X, y)
    intercept, slope = sklearn_model.intercept_, sklearn_model.coef_[0]
    
    print("Calculation of Regression Line:\n")
    print(f"Intercept is: {intercept}")
    print(f"Slope is: {slope}")

    # Fit the model using statsmodels to get p-values and R-squared
    statsmodels_model = fit_statsmodels_ols(X, y)
  #  print(statsmodels_model.summary())

    # Extract R-squared and p-values
    r_squared = statsmodels_model.rsquared
    p_values = statsmodels_model.pvalues

    print(f"R-squared: {r_squared}")
    #print(f"P-values: {p_values}")

    # Extracting specific p-values by index
    #  intercept_p_value = p_values[0]  # First p-value (intercept)
    slope_p_value = p_values[1]  # Second p-value (YearsExperience)

    #print(f"Intercept p-value: {intercept_p_value}")
    print(f"p-value (YearsExperience): {slope_p_value}")

    print("\nThe p-value is the probability of observing a t-statistic as extreme as, or more extreme than, the one calculated from your sample data, under the assumption that the null hypothesis is true.") 
    print("This is obtained from the t-distribution with n−2 degrees of freedom ")
    print("where n is the number of observations\n")

    if slope_p_value > 0.05:
        print("P-value is not signficant and therefore we accept the null hypothesis")
    if slope_p_value < 0.05:
        print("P-value is less than 0.05 and therefore we reject the null hypothesis. This means there is strong evidence that the predictor 𝑋 has a statistically significant effect on the outcome 𝑌")
    # Plotting the regression line
    plot_regression_line(df, intercept, slope)

    # Fit a linear regression line using scipy.stats (for comparison)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['YearsExperience'], df['Salary'])
    
    # plt.text(df['YearsExperience'].min(), df['Salary'].max(), f'y = {slope:.2f}x + {intercept:.2f}', ha='left')

if __name__ == "__main__":
    main()
