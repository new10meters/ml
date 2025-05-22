import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Step 1: Generate a synthetic dataset (e.g., a quadratic dataset with noise)
#np.random.seed(42)
# Generate random data
X = np.linspace(0, 1, 100)  # 100 points between 0 and 1
y = 2 * X**2 + 0.3 * np.random.randn(100)  # Quadratic data with noise

# Reshape X to match the required format for the algorithm
X = X[:, np.newaxis]

# Step 2: Define Locally Weighted Regression (LWR) function
def locally_weighted_regression(X_train, y_train, tau, X_test):
    m, n = X_train.shape
    y_pred = np.zeros(len(X_test))
    
    for i, x in enumerate(X_test):
        # Compute the weights for each point
        weights = np.exp(-np.sum((X_train - x)**2, axis=1)/ (2 * tau**2))
        
        # Construct the weighted design matrix
        W = np.diag(weights)
        
        # Solve for the parameters (theta) using the normal equation: (X'W*X)θ = X'W*y
        X_TW_X = X_train.T @ W @ X_train
        X_TW_y = X_train.T @ W @ y_train
        theta = inv(X_TW_X) @ X_TW_y
        
        # Prediction for the test point
        y_pred[i] = x @ theta
    
    return y_pred

# Step 3: Fit the model and make predictions
tau_values = [0.1, 0.3, 0.5]  # Different values of tau (bandwidth)
X_test = np.linspace(0, 1, 200)[:, np.newaxis]  # Test points for prediction

# Step 4: Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', label='Data points')

for tau in tau_values:
    y_pred = locally_weighted_regression(X, y, tau, X_test)
    plt.plot(X_test, y_pred, label=f'Tau = {tau}')

plt.title("Locally Weighted Regression (LWR)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
 