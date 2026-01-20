# Import pandas library
import pandas as pd

# Two-dimensional list (rows and columns)
data = [
    [101, "Amit", 85],
    [102, "Neha", 92],
    [103, "Raj", 78]
]

# Create DataFrame with column names
df = pd.DataFrame(data, columns=["Roll No", "Name", "Marks"])

# Display the DataFrame
print(df)

