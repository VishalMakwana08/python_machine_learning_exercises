import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Create a sample dataset with categorical values
df = pd.DataFrame({
    "City": ["Surat", "Ahmedabad", "Vadodara", "Surat", "Rajkot"],
    "Grade": ["A", "B", "A", "C", "B"]
})

print("Original Dataset:")
print(df)

# Step 2: Create LabelEncoder objects
city_encoder = LabelEncoder()
grade_encoder = LabelEncoder()

# Step 3: Apply label encoding (convert text categories into numbers)
df["City_Encoded"] = city_encoder.fit_transform(df["City"])
df["Grade_Encoded"] = grade_encoder.fit_transform(df["Grade"])

print("\nDataset After Label Encoding:")
print(df)

# Step 4: Show mapping (which category became which number)
print("\nCity Encoding Mapping:")
for i, name in enumerate(city_encoder.classes_):
    print(name, "->", i)

print("\nGrade Encoding Mapping:")
for i, name in enumerate(grade_encoder.classes_):
    print(name, "->", i)

