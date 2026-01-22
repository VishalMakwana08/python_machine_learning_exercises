import numpy as np
from sklearn.linear_model import LinearRegression

# Step 1: Create sample X values
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)   # must be 2D for sklearn

# Step 2: Create different relationships (different y)
y_linear = np.array([2, 4, 6, 8, 10])          # y = 2x (linear)
y_increasing = np.array([3, 5, 7, 9, 11])      # y = 2x + 1 (linear)
y_random = np.array([2, 5, 5, 9, 12])          # not perfectly linear

# Function to train and show results
def run_linear_regression(X, y, name):
    model = LinearRegression()
    model.fit(X, y)

    print("\nRelationship:", name)
    print("Slope (m):", model.coef_[0])
    print("Intercept (c):", model.intercept_)
    print("Equation: y =", model.coef_[0], "* x +", model.intercept_)

    y_pred = model.predict(X)
    print("Predicted values:", y_pred)

# Step 3: Run models for different relationships
run_linear_regression(X, y_linear, "Perfect Linear (y = 2x)")
run_linear_regression(X, y_increasing, "Linear with Intercept (y = 2x + 1)")
run_linear_regression(X, y_random, "Not Perfect Linear (approx fit)")



