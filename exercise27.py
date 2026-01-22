import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Step 1: Load regression dataset (Diabetes)
data = load_diabetes()
X = data.data
y = data.target  # continuous target (regression)

# Step 2: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Step 3: Train a regression model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Step 4: Predict continuous values
y_pred = reg_model.predict(X_test)

# Step 5: Convert continuous values into classes (0/1) using median threshold
threshold = np.median(y_train)

# Actual classes (from y_test)
y_test_class = (y_test >= threshold).astype(int)

# Predicted classes (from y_pred)
y_pred_class = (y_pred >= threshold).astype(int)

# Step 6: Confusion matrix + accuracy
cm = confusion_matrix(y_test_class, y_pred_class)
acc = accuracy_score(y_test_class, y_pred_class)

print("Median Threshold used:", threshold)

print("\nConfusion Matrix (for converted classes):")
print(cm)

print("\nAccuracy:", round(acc, 3))

print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class))

