import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Create a simple dataset
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11, 13, 15, 18, 19, 21])  # almost linear (a little variation)

# Step 2: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Step 3: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict on test data
y_pred = model.predict(X_test)

# Step 5: Evaluate using different metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Test Actual y values :", y_test)
print("Test Predicted values:", np.round(y_pred, 2))

print("\nEvaluation Metrics:")
print("1) MAE  (Mean Absolute Error)      =", round(mae, 3))
print("2) MSE  (Mean Squared Error)       =", round(mse, 3))
print("3) RMSE (Root Mean Squared Error)  =", round(rmse, 3))
print("4) R2 Score (Coefficient of Determination) =", round(r2, 3))

