import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    "Name": ["Amit", "Neha", "Raj"],
    "Age": [20, 21, 22],
    "City": ["Surat", "Ahmedabad", "Vadodara"]
})

print("Original DataFrame:\n", df)

# 1) Change column names
df.columns = ["Student_Name", "Student_Age", "Student_City"]

# OR (another method)
# df.rename(columns={"Name": "Student_Name", "Age": "Student_Age", "City": "Student_City"}, inplace=True)

# 2) Change row indexes
df.index = ["S1", "S2", "S3"]

# OR (another method)
# df.rename(index={0: "S1", 1: "S2", 2: "S3"}, inplace=True)

print("\nUpdated DataFrame:\n", df)
