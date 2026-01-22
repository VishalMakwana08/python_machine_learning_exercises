import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Breast Cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target  # 0 = malignant, 1 = benign

print("Dataset: Breast Cancer")
print("Total samples:", X.shape[0])
print("Total features:", X.shape[1])
print("Classes:", data.target_names)  # ['malignant', 'benign']

# Step 2: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Step 3: Feature scaling (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Build SVM model
# kernel='linear' makes it simple and easy to understand
model = SVC(kernel="linear", random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Predict on test set
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate model
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", round(acc, 3))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))

# Step 7: Predict for one new sample (example: take first test row)
sample = X_test_scaled[0].reshape(1, -1)
pred = model.predict(sample)[0]

print("\nExample Prediction for one sample:")
print("Predicted class:", data.target_names[pred])

