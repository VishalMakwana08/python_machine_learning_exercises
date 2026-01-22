import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

# Step 1: Create sample data (3 groups)
np.random.seed(42)
c1 = np.random.normal(loc=[2, 2], scale=0.6, size=(30, 2))
c2 = np.random.normal(loc=[6, 6], scale=0.7, size=(30, 2))
c3 = np.random.normal(loc=[10, 2], scale=0.6, size=(30, 2))
X = np.vstack((c1, c2, c3))

print("Total data points:", X.shape[0])

# Step 2: Scale data (recommended for distance-based clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Create linkage matrix (Agglomerative clustering info)
# method='ward' is commonly used with Euclidean distance
Z = linkage(X_scaled, method="ward")

print("\nLinkage Matrix (first 10 rows):")
print(Z[:10])

# Step 4: Plot dendrogram using the linkage matrix
plt.figure(figsize=(8, 4))
dendrogram(Z)
plt.title("Dendrogram (from Linkage Matrix)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
