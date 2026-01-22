import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Create a sample dataset (numeric columns for correlation)
df = pd.DataFrame({
    "Age": [18, 20, 22, 24, 26, 28, 30, 32],
    "StudyHours": [1, 2, 2, 3, 4, 4, 5, 6],
    "SleepHours": [8, 7, 7, 6, 6, 5, 5, 4],
    "Marks": [40, 50, 55, 65, 75, 80, 88, 95]
})

print("Dataset:\n")
print(df)

# Step 2: Find correlation matrix
corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n")
print(corr.round(2))

# Step 3: Plot heatmap (using matplotlib only)
plt.figure(figsize=(6, 5))
plt.imshow(corr, aspect='auto')
plt.colorbar()

# Step 4: Add labels (feature names) on axes
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

# Step 5: Write correlation values inside each cell (for clarity)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
