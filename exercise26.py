import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load diabetes dataset
data = load_diabetes()
X = data.data
y_continuous = data.target

# Step 2: Convert continuous target to binary classes for Logistic Regression
# If target >= median -> 1 else 0
threshold = np.median(y_continuous)
y = (y_continuous >= threshold).astype(int)

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train logistic regression model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# Step 6: Predict test data
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate using accuracy metric
acc = accuracy_score(y_test, y_pred)

print("Accuracy of Logistic Regression Model:", round(acc, 3))
print("\nTotal Test Samples:", len(y_test))
print("Correct Predictions:", np.sum(y_test == y_pred))
print("Wrong Predictions:", np.sum(y_test != y_pred))

