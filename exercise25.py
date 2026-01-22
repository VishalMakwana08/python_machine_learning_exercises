import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Diabetes dataset (from scikit-learn)
data = load_diabetes()
X = data.data
y_continuous = data.target  # this is continuous (regression target)

print("Diabetes dataset loaded")
print("X shape:", X.shape)
print("Target shape:", y_continuous.shape)

# Step 2: Convert continuous target into binary classes for Logistic Regression
# Rule: if target >= median => 1 (high), else 0 (low)
threshold = np.median(y_continuous)
y = (y_continuous >= threshold).astype(int)

print("\nConverted target to binary classes using median threshold:")
print("Threshold (median) =", threshold)
print("Class 0 count =", np.sum(y == 0))
print("Class 1 count =", np.sum(y == 1))

# Step 3: Split dataset into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Step 4: Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Build and train Logistic Regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", round(accuracy_score(y_test, y_pred), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

