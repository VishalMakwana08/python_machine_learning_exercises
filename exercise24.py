import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Generate data from Standard Normal Distribution (mean=0, std=1)
# X will have 2 features
np.random.seed(42)
X = np.random.normal(loc=0, scale=1, size=(500, 2))

# Step 2: Create labels (simple classification rule)
# If (x1 + x2) > 0 => class 1 else class 0
y = (X[:, 0] + X[:, 1] > 0).astype(int)

print("First 5 data points (X):")
print(X[:5])
print("\nFirst 5 labels (y):")
print(y[:5])

# Step 3: Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Step 4: Standardize features (mean=0, std=1) using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train a simple classifier (Logistic Regression)
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



