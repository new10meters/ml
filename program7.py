import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.metrics import mean_squared_error, r2_score  


import warnings 
warnings.filterwarnings("ignore")

def perform_linear_regression(csv_file_path):  
    # Load the dataset from CSV file  
    data = pd.read_csv(csv_file_path)  
   
    
    X = data.iloc[:, :-1]  # Features (all columns except the last one)  
    y = data.iloc[:, -1]   # Target variable (last column)  

    # Split the data into training and testing sets  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

    # Create a linear regression model  
    linear_reg = LinearRegression()  

    # Train the model  
    linear_reg.fit(X_train, y_train)  

    # Predict on the test set  
    y_pred = linear_reg.predict(X_test)  

    # Evaluate the model  
    mse = mean_squared_error(y_test, y_pred)  
    r2 = r2_score(y_test, y_pred)  

    # Print results for Linear Regression  
    print(f"Linear Regression - Mean Squared Error: {mse:.2f}")  
    print(f"Linear Regression - R-squared: {r2:.2f}")  

    # Plotting predicted vs actual values  
    plt.figure(figsize=(10, 6))  
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')  
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Perfect Fit')  
    plt.xlabel('Actual Values')  
    plt.ylabel('Predicted Values')  
    plt.title('Linear Regression - Predicted vs Actual Values (Boston Housing)')  
    plt.legend()  
    plt.show()  

def perform_polynomial_regression(csv_file_path):  
    # Load the Auto MPG dataset  
    data = pd.read_csv(csv_file_path)  


    # Handle missing values: Remove rows with missing 'mpg' values (target variable)  
    data = data.dropna(subset=['mpg'])  

    # Select features and target variable  
    X = data[['horsepower']]  # Selecting 'horsepower' explicitly to maintain as DataFrame  
    y = data['mpg']  

    # Convert 'horsepower' to numeric (if necessary)  
    X.loc[:, 'horsepower'] = pd.to_numeric(X['horsepower'], errors='coerce')  

    # Drop any rows with NaN values created during conversion (if any)  
    X = X.dropna()  
    y = y.loc[X.index]  # Ensure that the target (y) aligns with the feature data (X)  

    # Split the dataset into training and testing sets  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

    degree = 2  
    # Create polynomial features  
    poly = PolynomialFeatures(degree)  # Change degree as needed  

    # Transform the features into polynomial features  
    X_train_poly = poly.fit_transform(X_train)  
    X_test_poly = poly.transform(X_test)  

    # Create a linear regression model  
    linear_reg = LinearRegression()  

    # Train the model on polynomial features  
    linear_reg.fit(X_train_poly, y_train)  

    # Predict on the test set  
    y_pred = linear_reg.predict(X_test_poly)  

    # Evaluate the model  
    mse = mean_squared_error(y_test, y_pred)  
    r2 = r2_score(y_test, y_pred)  

    # Print results for Polynomial Regression  
    print(f"Polynomial Regression - Mean Squared Error: {mse:.2f}")  
    print(f"Polynomial Regression - R-squared: {r2:.2f}")  

    # Plotting  
    plt.scatter(X['horsepower'], y, color='blue', label='Data')  # Use 'horsepower' as x-axis  
    X_range = np.linspace(X['horsepower'].min(), X['horsepower'].max(), 100).reshape(-1, 1)   
    X_range_poly = poly.transform(X_range)  # Ensure it uses polynomial features correctly  
    y_range_pred = linear_reg.predict(X_range_poly)  
    plt.plot(X_range, y_range_pred, color='red', label='Polynomial Fit')  
    plt.xlabel('Horsepower')  
    plt.ylabel('MPG')   
    plt.legend()  
    plt.title(f'Polynomial Regression (degree {degree})')  
    plt.show()  



perform_linear_regression('BostonHousing.csv')  # Replace with actual path to the CSV file  
perform_polynomial_regression('auto-mpg.csv')  # Replace with actual path to the CSV file  