import pandas as pd

# Step 1: Create a sample dataset with categorical columns
df = pd.DataFrame({
    "City": ["Surat", "Ahmedabad", "Vadodara", "Surat", "Rajkot"],
    "Grade": ["A", "B", "A", "C", "B"]
})

print("Original Dataset:")
print(df)

# Step 2: Apply One-Hot Encoding using pandas get_dummies()
encoded_df = pd.get_dummies(df, columns=["City", "Grade"])

print("\nDataset After One-Hot Encoding:")
print(encoded_df)



