import pandas as pd

# Step 1: Create a sample dataset (DataFrame)
df = pd.DataFrame({
    "Name": ["Amit", "Neha", "Raj", "Priya", "Karan"],
    "Age": [20, 21, 22, 20, 23],
    "Marks": [78, 85, 69, 92, 74]
})

print("Dataset:\n")
print(df)

# Step 2: Summary operations on numeric columns
print("\nSummary Operations (Age and Marks):")

print("\n1) Count (non-missing values):")
print(df[["Age", "Marks"]].count())

print("\n2) Sum:")
print(df[["Age", "Marks"]].sum())

print("\n3) Mean (Average):")
print(df[["Age", "Marks"]].mean())

print("\n4) Min:")
print(df[["Age", "Marks"]].min())

print("\n5) Max:")
print(df[["Age", "Marks"]].max())

print("\n6) Standard Deviation:")
print(df[["Age", "Marks"]].std())

# Step 3: One-line complete summary (very common in data analysis)
print("\n7) Full Summary using describe():")
print(df[["Age", "Marks"]].describe())