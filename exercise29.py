import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# Step 1: Create a simple 2D dataset (two features so we can visualize)
np.random.seed(42)

# Class 0 points
X0 = np.random.normal(loc=[2, 2], scale=0.6, size=(60, 2))
y0 = np.zeros(60)

# Class 1 points
X1 = np.random.normal(loc=[6, 6], scale=0.6, size=(60, 2))
y1 = np.ones(60)

# Combine both classes
X = np.vstack((X0, X1))
y = np.hstack((y0, y1))

# Step 2: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Step 3: Normalize data using Min-Max Normalization (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_norm = scaler.fit_transform(X_train)
X_test_norm = scaler.transform(X_test)

# Step 4: Train a simple classifier (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_norm, y_train)

# Step 5: Plot training set results
plt.figure()
plt.scatter(X_train_norm[:, 0], X_train_norm[:, 1], c=y_train)
plt.title("Training Set (Normalized) - Class Distribution")
plt.xlabel("Feature 1 (Normalized)")
plt.ylabel("Feature 2 (Normalized)")
plt.show()

# Step 6: Plot test set results
plt.figure()
plt.scatter(X_test_norm[:, 0], X_test_norm[:, 1], c=y_test)
plt.title("Test Set (Normalized) - Class Distribution")
plt.xlabel("Feature 1 (Normalized)")
plt.ylabel("Feature 2 (Normalized)")
plt.show()
