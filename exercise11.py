import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Create a sample dataset (features have different ranges)
df = pd.DataFrame({
    "Age": [18, 20, 22, 24, 26],
    "Salary": [20000, 25000, 30000, 35000, 40000]
})

print("Original Dataset:")
print(df)

# Step 2: Create MinMaxScaler object (Normalization)
scaler = MinMaxScaler(feature_range=(0, 1))

# Step 3: Apply normalization (fit + transform)
normalized_values = scaler.fit_transform(df)

# Step 4: Convert normalized result back to DataFrame
normalized_df = pd.DataFrame(normalized_values, columns=df.columns)

print("\nDataset After Normalization (Min-Max Scaling):")
print(normalized_df.round(3))

