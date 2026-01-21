import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Create a sample dataset (different feature ranges)
df = pd.DataFrame({
    "Age": [18, 20, 22, 24, 26],
    "Salary": [20000, 25000, 30000, 35000, 40000]
})

print("Original Dataset:")
print(df)

# Step 2: Create StandardScaler object
scaler = StandardScaler()

# Step 3: Apply standardization (fit + transform)
scaled_values = scaler.fit_transform(df)

# Step 4: Convert scaled result back to DataFrame
scaled_df = pd.DataFrame(scaled_values, columns=df.columns)

print("\nDataset After Standardization (Feature Scaling):")
print(scaled_df.round(3))

