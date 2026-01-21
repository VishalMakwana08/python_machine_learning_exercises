import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create sample data (one feature column)
X = np.array([[10], [20], [30], [40], [50]], dtype=float)

print("Original Data (X):")
print(X)

# Step 2: Scale the data into a fixed range (0 to 1)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

print("\nScaled Data (0 to 1 range):")
print(X_scaled)

# Step 3: Generate datapoints in a range using linspace
# This creates evenly spaced values between start and end
points = np.linspace(0, 1, 6)  # 6 points from 0 to 1

print("\nGenerated datapoints in range 0 to 1:")
print(points)

