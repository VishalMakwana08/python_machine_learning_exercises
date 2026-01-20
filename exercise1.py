import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    "Number": [1, 2, 3, 4, 5]
})

# Empty list to store values for the new column
squares = []

# Use for loop to calculate square of each number
for n in df["Number"]:
    squares.append(n * n)

# Add the list as a new column in DataFrame
df["Square"] = squares

# Display the updated DataFrame
print(df)

