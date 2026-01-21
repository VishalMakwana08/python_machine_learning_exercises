import numpy as np
from sklearn import preprocessing

# Step 1: Create a sample dataset (2D array)
# Each column is a feature
X = np.array([
    [10, 20, 30],
    [20, 30, 40],
    [30, 40, 50],
    [40, 50, 60]
], dtype=float)

print("Original Data (X):")
print(X)

# Step 2: Perform mean removal (center the data)
# This subtracts the mean of each column from that column
X_mean_removed = preprocessing.scale(X, with_mean=True, with_std=False)

print("\nData After Mean Removal:")
print(X_mean_removed)

# Step 3: Show mean of each column before and after
print("\nColumn means before mean removal:")
print(np.mean(X, axis=0))

print("\nColumn means after mean removal (should be ~0):")
print(np.mean(X_mean_removed, axis=0))

