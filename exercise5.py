import pandas as pd

# Step 1: Create a sample dataset (DataFrame)
df = pd.DataFrame({
    "RollNo": [101, 102, 103, 104, 105],
    "Name": ["Amit", "Neha", "Raj", "Priya", "Karan"],
    "Age": [20, 21, 22, 20, 23],
    "City": ["Surat", "Ahmedabad", "Vadodara", "Rajkot", "Bhavnagar"],
    "Marks": [78, 85, 69, 92, 74]
})

print("Original Dataset:\n")
print(df)

# Step 2: Extract specified rows (by index) and specified columns (by names)
# Example: extract rows 1 to 3 (index 1,2,3) and columns Name, City, Marks
extracted = df.loc[1:3, ["Name", "City", "Marks"]]

print("\nExtracted rows (1 to 3) and columns (Name, City, Marks):\n")
print(extracted)
