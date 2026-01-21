import numpy as np
from sklearn.preprocessing import Binarizer

# Step 1: Create sample data (vector)
X = np.array([[-2.5], [-0.5], [0.0], [0.5], [2.5]])

print("Original Vector (X):")
print(X)

# Step 2: Create Binarizer with a threshold
# Rule:
# value > threshold  -> 1
# value <= threshold -> 0
binarizer = Binarizer(threshold=0.0)

# Step 3: Apply binarization
X_binary = binarizer.fit_transform(X)

print("\nBinary Vector after Binarization (threshold=0.0):")
print(X_binary)

