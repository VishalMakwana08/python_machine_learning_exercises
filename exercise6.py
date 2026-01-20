import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Step 1: Create a sample dataset with missing values (NaN)
df = pd.DataFrame({
    "Age": [20, 21, np.nan, 23, np.nan],
    "Marks": [78, np.nan, 69, 92, 74]
})

print("Original Dataset (with missing values):")
print(df)

# Step 2: Create Imputer with mean strategy
imputer = SimpleImputer(strategy="mean")

# Step 3: Fit and transform the data (replace NaN with column mean)
filled_data = imputer.fit_transform(df)

# Step 4: Convert back to DataFrame for easy display
df_filled = pd.DataFrame(filled_data, columns=df.columns)

print("\nDataset After Handling Missing Values (Mean Strategy):")
print(df_filled)

# Step 5: Show the means used
print("\nMeans used to fill missing values:")
print("Age mean  =", imputer.statistics_[0])
print("Marks mean=", imputer.statistics_[1])
