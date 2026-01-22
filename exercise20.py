import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset
df = pd.DataFrame({
    "Name": ["Amit", "Neha", "Raj", "Priya", "Karan", "Meera", "Arjun", "Riya"],
    "Age":  [20, 21, 22, 20, 23, 22, 21, 24],
    "Marks":[78, 85, 69, 92, 74, 88, 65, 95]
})

print("Dataset:\n")
print(df)

print("\nGraphs saved as:")
print("1) bar_marks.png")
print("2) hist_age.png")
print("3) scatter_age_marks.png")
print("4) line_marks_trend.png")

# 1) Bar chart
plt.figure()
plt.bar(df["Name"], df["Marks"])
plt.title("Marks of Each Student")
plt.xlabel("Student Name")
plt.ylabel("Marks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_marks.png")
plt.show()

# 2) Histogram
plt.figure()
plt.hist(df["Age"], bins=5)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("hist_age.png")
plt.show()

# 3) Scatter plot
plt.figure()
plt.scatter(df["Age"], df["Marks"])
plt.title("Age vs Marks")
plt.xlabel("Age")
plt.ylabel("Marks")
plt.tight_layout()
plt.savefig("scatter_age_marks.png")
plt.show()

# 4) Line chart
plt.figure()
plt.plot(df["Name"], df["Marks"], marker="o")
plt.title("Marks Trend")
plt.xlabel("Student Name")
plt.ylabel("Marks")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("line_marks_trend.png")
plt.show()
