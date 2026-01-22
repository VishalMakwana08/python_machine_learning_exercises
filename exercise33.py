import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Create different data points (3 groups)
np.random.seed(42)

# Cluster 1 points
c1 = np.random.normal(loc=[2, 2], scale=0.5, size=(60, 2))

# Cluster 2 points
c2 = np.random.normal(loc=[6, 6], scale=0.5, size=(60, 2))

# Cluster 3 points
c3 = np.random.normal(loc=[10, 2], scale=0.5, size=(60, 2))

# Combine all points into one dataset
X = np.vstack((c1, c2, c3))

# Step 2: Scale the dataset (recommended for K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Step 4: Get cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Cluster Centers (in scaled space):")
print(centers)

# Step 5: Plot data points and cluster centers
plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)

# Plot cluster centers with a different marker
plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=250)

plt.title("K-Means Clustering: Data Points and Cluster Centers")
plt.xlabel("Feature 1 (Standardized)")
plt.ylabel("Feature 2 (Standardized)")
plt.show()
