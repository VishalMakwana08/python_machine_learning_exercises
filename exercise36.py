import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Step 1: Create sample data points (3 natural groups)
np.random.seed(42)

c1 = np.random.normal(loc=[2, 2], scale=0.6, size=(70, 2))
c2 = np.random.normal(loc=[6, 6], scale=0.7, size=(70, 2))
c3 = np.random.normal(loc=[10, 2], scale=0.6, size=(70, 2))

X = np.vstack((c1, c2, c3))

print("Total data points:", X.shape[0])

# Step 2: Scale the data (distance-based clustering works better)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Build Agglomerative Clustering model
# n_clusters=3 because we created 3 groups
model = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels = model.fit_predict(X_scaled)

print("\nCluster labels for first 10 points:")
print(labels[:10])

print("\nTotal clusters found:", len(np.unique(labels)))

# Step 4: Visualize clusters
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title("Agglomerative Clustering (Hierarchical)")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.show()
