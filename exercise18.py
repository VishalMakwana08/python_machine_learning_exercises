import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Step 1: Create Advertising Sales dataset (TV, Radio, Newspaper -> Sales)
data = {
    "TV": [230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6, 199.8],
    "Radio": [37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1, 2.6],
    "Newspaper": [69.2, 45.1, 69.3, 58.5, 12.8, 75.0, 23.5, 11.6, 1.0, 21.2],
    "Sales": [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8, 10.6]
}

df = pd.DataFrame(data)

print("Advertising Dataset (first 5 rows):")
print(df.head())

# Step 2: Split features (X) and target (y)
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test)

# Step 6: Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Coefficients:")
print("TV coefficient       :", model.coef_[0])
print("Radio coefficient    :", model.coef_[1])
print("Newspaper coefficient:", model.coef_[2])
print("Intercept            :", model.intercept_)

print("\nTest Actual Sales   :", list(y_test.values))
print("Test Predicted Sales:", list(np.round(y_pred, 2)))

print("\nEvaluation Metrics:")
print("MAE  =", round(mae, 3))
print("MSE  =", round(mse, 3))
print("RMSE =", round(rmse, 3))
print("R2   =", round(r2, 3))

